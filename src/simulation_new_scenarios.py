"""Simulation study for the additional scenarios comparing the two methods.

This script complements the simulation in simulation_and_frechet_method.ipynb
(scenario 1, linear in the space of quantile functions, differing supports) with
three scenarios that vary which method is correctly specified:

- Scenario 2 ("transport"): sine-perturbed uniform distributions with common
  support [0, 1]. The truth is linear in qf space, so global Fréchet regression
  is correctly specified, but the LQD method no longer has to estimate the
  location of the support. This isolates how much of the gap in scenario 1 is
  due to the support estimation.
- Scenario 3 ("lqd_linear"): the log quantile densities follow a functional
  linear model in the predictor, so the LQD method is correctly specified and
  global Fréchet regression is misspecified.
- Scenario 4 ("nonlinear"): location-scale normal distributions whose location
  is sinusoidal and whose scale is quadratic in the predictor; both methods are
  misspecified.

All scenarios use directly observed densities (no density estimation step).
Results are stored in sim_results/stored_ise_new_scenarios.pkl and a table of
mean ISE values is printed.

"""

import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from config import SRC
from tools.data_generation_tools import (
    gen_lqd_linear_qfs_regression,
    gen_params_nonlinear_regression,
    gen_predictor_values_regression,
    gen_transport_qfs_regression,
    gen_y_qf,
    lqd_linear_qfs,
    transport_qfs,
)
from tools.frechet_tools import ise_wasserstein, solve_frechet_qp
from tools.function_tools import (
    inverse_log_qd_transform,
    log_qd_transform,
)

# Number of Monte Carlo repetitions
M_REPS = 200
# Sample sizes to compare
SAMPLE_SIZES = [50, 100, 200]
# Fineness of grid on which functions are evaluated
GRID_SIZE = 500
U_GRID = np.linspace(0, 1, GRID_SIZE)
# Predictor values for which conditional distributions are predicted
PREDICTOR_BOUNDS = (-1, 1)
GRID_TO_PREDICT_SIZE = 50
X_GRID = np.linspace(PREDICTOR_BOUNDS[0], PREDICTOR_BOUNDS[1], GRID_TO_PREDICT_SIZE)

# Scenario 2: common-support transport maps
TRANSPORT_ALPHA = 0.5
TRANSPORT_NOISE = 0.25
# Scenario 3: linear model in LQD space
LQD_C0 = 0.0
LQD_NOISE_SD = 0.1
# Scenario 4: nonlinear location-scale process (noise structure as in scenario 1)
MU_PARAMS = {"mu0": 0, "beta": 4, "v1": 0.25}
SIGMA_PARAMS = {"sigma0": 3, "gamma": 2, "v2": 2}


def _resolve_n_jobs(n_jobs):
    """Normalize the requested number of worker processes."""
    if n_jobs is None or n_jobs == -1:
        return os.cpu_count() or 1
    return max(1, int(n_jobs))


def frechet_ise(qfs, predictor_vals, true_qfs):
    """ISE of global Fréchet regression against the true conditional qfs."""
    estimates = solve_frechet_qp(
        xs_to_predict=X_GRID,
        x_observed=predictor_vals,
        quantile_functions=qfs,
    )
    return ise_wasserstein(estimates, true_qfs, X_GRID, already_qf=True)


def lqd_ise(qfs, predictor_vals, true_qfs):
    """ISE of functional regression with LQD method against the true qfs."""
    pdfs = [qf.drop_inf().invert().differentiate() for qf in qfs]
    lqdfs, start_vals = log_qd_transform(pdfs, different_supports=True)
    predictor_matrix = np.array(
        (np.ones_like(predictor_vals), predictor_vals),
    ).transpose()
    log_betahat = (
        np.linalg.inv(predictor_matrix.transpose() @ predictor_matrix)
        @ predictor_matrix.transpose()
        @ lqdfs
    )
    lqdf_hat = X_GRID * log_betahat[1] + log_betahat[0]
    interpolated_start_vals = np.interp(X_GRID, predictor_vals, start_vals)
    pdf_hat = inverse_log_qd_transform(lqdf_hat, interpolated_start_vals)
    qf_hat = [pdf.integrate().invert() for pdf in pdf_hat]
    return ise_wasserstein(qf_hat, true_qfs, X_GRID, already_qf=True)


def gen_scenario_sample(scenario, predictor_vals, seed):
    """Draw a sample of conditional qfs for a given scenario."""
    if scenario == "transport":
        return gen_transport_qfs_regression(
            predictor_vals,
            U_GRID,
            TRANSPORT_ALPHA,
            TRANSPORT_NOISE,
            seed,
        )
    if scenario == "lqd_linear":
        return gen_lqd_linear_qfs_regression(
            predictor_vals,
            U_GRID,
            LQD_C0,
            LQD_NOISE_SD,
            seed,
        )
    if scenario == "nonlinear":
        mus, sigmas = gen_params_nonlinear_regression(
            MU_PARAMS,
            SIGMA_PARAMS,
            predictor_vals,
            seed,
        )
        return [qf.drop_inf() for qf in gen_y_qf(mus, sigmas, U_GRID)]
    msg = f"Unknown scenario: {scenario}"
    raise ValueError(msg)


def true_scenario_qfs(scenario):
    """True conditional qfs of a scenario on the prediction grid."""
    if scenario == "transport":
        return transport_qfs(TRANSPORT_ALPHA * X_GRID, U_GRID)
    if scenario == "lqd_linear":
        return lqd_linear_qfs(X_GRID, U_GRID, LQD_C0)
    if scenario == "nonlinear":
        true_mus = MU_PARAMS["mu0"] + MU_PARAMS["beta"] * np.sin(np.pi * X_GRID)
        true_sigmas = SIGMA_PARAMS["sigma0"] + SIGMA_PARAMS["gamma"] * X_GRID**2
        return [qf.drop_inf() for qf in gen_y_qf(true_mus, true_sigmas, U_GRID)]
    msg = f"Unknown scenario: {scenario}"
    raise ValueError(msg)


def _run_replication(scenario, i):
    """Run one Monte Carlo replication for all sample sizes."""
    true_qfs = true_scenario_qfs(scenario)
    frechet_values = np.zeros(len(SAMPLE_SIZES))
    lqd_values = np.zeros(len(SAMPLE_SIZES))

    for j, n in enumerate(SAMPLE_SIZES):
        seed = 10_000 * (i + 1) + n  # unique seed per rep and sample size
        predictor_vals = gen_predictor_values_regression(
            n,
            PREDICTOR_BOUNDS,
            seed,
        )
        qfs = gen_scenario_sample(scenario, predictor_vals, seed)
        frechet_values[j] = frechet_ise(qfs, predictor_vals, true_qfs)
        lqd_values[j] = lqd_ise(qfs, predictor_vals, true_qfs)

    return scenario, i, frechet_values, lqd_values


def _run_simulation_sequential(scenarios, progress_every):
    """Run the Monte Carlo study in the original sequential order."""
    results = {
        scenario: {
            "frechet": np.zeros((M_REPS, len(SAMPLE_SIZES))),
            "lqd": np.zeros((M_REPS, len(SAMPLE_SIZES))),
        }
        for scenario in scenarios
    }
    for scenario in scenarios:
        start = time.time()
        for i in range(M_REPS):
            _, _, frechet_values, lqd_values = _run_replication(scenario, i)
            results[scenario]["frechet"][i] = frechet_values
            results[scenario]["lqd"][i] = lqd_values
            if (i + 1) % progress_every == 0 or i + 1 == M_REPS:
                elapsed = time.time() - start
                print(  # noqa: T201
                    f"{scenario}: rep {i + 1}/{M_REPS} done "
                    f"({elapsed:.0f}s elapsed)",
                    flush=True,
                )
    return results


def run_simulation(
    scenarios=("transport", "lqd_linear", "nonlinear"),
    n_jobs=None,
    progress_every=10,
):
    """Run the Monte Carlo study and return ISE arrays per scenario and method."""
    n_jobs = _resolve_n_jobs(n_jobs)
    if n_jobs == 1:
        return _run_simulation_sequential(scenarios, progress_every)

    results = {
        scenario: {
            "frechet": np.zeros((M_REPS, len(SAMPLE_SIZES))),
            "lqd": np.zeros((M_REPS, len(SAMPLE_SIZES))),
        }
        for scenario in scenarios
    }
    completed = dict.fromkeys(scenarios, 0)
    start = time.time()
    jobs = [(scenario, i) for scenario in scenarios for i in range(M_REPS)]

    print(f"Running with {n_jobs} worker processes")  # noqa: T201
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [
            executor.submit(_run_replication, scenario, i)
            for scenario, i in jobs
        ]
        for future in as_completed(futures):
            scenario, i, frechet_values, lqd_values = future.result()
            results[scenario]["frechet"][i] = frechet_values
            results[scenario]["lqd"][i] = lqd_values
            completed[scenario] += 1
            if (
                completed[scenario] % progress_every == 0
                or completed[scenario] == M_REPS
            ):
                elapsed = time.time() - start
                print(  # noqa: T201
                    f"{scenario}: {completed[scenario]}/{M_REPS} reps done "
                    f"({elapsed:.0f}s elapsed)",
                    flush=True,
                )
    return results


def print_table(results):
    """Print mean ISE values in the layout of the table in the thesis."""
    header = "".join(f"n = {n:>5} " for n in SAMPLE_SIZES)
    print(f"\n{'Mean ISE':<28}{header}")  # noqa: T201
    for scenario, methods in results.items():
        for method, ises in methods.items():
            means = "".join(f"{val:>9.4f} " for val in ises.mean(axis=0))
            print(f"{scenario + ', ' + method:<28}{means}")  # noqa: T201


if __name__ == "__main__":
    simulation_results = run_simulation()
    file_path_pickle = SRC / "sim_results" / "stored_ise_new_scenarios.pkl"
    with file_path_pickle.open("wb") as handle:
        pickle.dump(simulation_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print_table(simulation_results)
