"""This module contains code for generating data from a truncated normal distribution,
and functions that generate parameters according to the three scenarios outlined in
Petersen and Mueller (2016)'s simulation study.
"""

import warnings

import numpy as np
from frechet_fda.function_class import Function
from frechet_fda.tools.kernel_methods import (
    boundary_corrected_density_estimator,
    density_estimator,
)
from frechet_fda.tools.numerics_tools import riemann_sum_cumulative
from scipy.stats import norm, truncnorm


def gen_predictor_values_regression(
    size: int = 100, pred_bounds: tuple = (-1, 1), seed: int = None,
):
    """Generates scalar predictor for Fréchet regression."""
    predictor = np.random.default_rng(seed).uniform(
        pred_bounds[0], pred_bounds[1], size,
    )
    return np.sort(predictor)


def gen_params_regression(
    mu_params: dict, sigma_params: dict, predictor: np.ndarray, seed: int = None,
):
    """Generates mus and sigmas for conditional distributions in regression context.

    This reflects the first simulation scenario of Petersen & Müller (2019). Need to
    pass mu_params and sigma params, which have to contain specific entries.

    """
    # Generate mus that depend linearly on x
    mus = np.random.default_rng(seed).normal(
        loc=mu_params["mu0"] + mu_params["beta"] * predictor, scale=mu_params["v1"],
    )
    # Generate sigmas that depend linearly on x
    sh = (
        sigma_params["sigma0"] + sigma_params["gamma"] * predictor
    ) ** 2 / sigma_params["v2"]
    sc = sigma_params["v2"] / (
        sigma_params["sigma0"] + sigma_params["gamma"] * predictor
    )
    sigmas = np.random.default_rng(seed).gamma(shape=sh, scale=sc)

    return mus, sigmas


def gen_params_regression_squared(
    mu_params: dict, sigma_params: dict, predictor: np.ndarray, seed: int = None,
):
    """Generates mus and sigmas for conditional distributions in regression context.

    This is a slightly modified version of the first simulation scenario of
    Petersen & Müller (2019). Need to pass mu_params and sigma params, which have to
    contain specific entries.

    """
    # Generate mus that depend quadratically on x
    mus = np.random.default_rng(seed).normal(
        loc=mu_params["mu0"]
        + mu_params["beta1"] * predictor
        + mu_params["beta2"] * predictor ** 2,
        scale=mu_params["v1"],
    )
    # Generate sigmas that depend quadratically on x
    sh = (
        sigma_params["sigma0"]
        + sigma_params["gamma1"] * predictor
        + sigma_params["gamma2"] * predictor ** 2
    ) ** 2 / sigma_params["v2"]
    sc = sigma_params["v2"] / (
        sigma_params["sigma0"]
        + sigma_params["gamma1"] * predictor
        + sigma_params["gamma2"] * predictor ** 2
    )
    sigmas = np.random.default_rng(seed).gamma(shape=sh, scale=sc)

    return mus, sigmas


def gen_y_qf(mu: np.ndarray, sigma: np.ndarray, eval_grid: np.ndarray):
    """Generate quantile function of Y_i given X_i."""
    ys = (
        mu[..., np.newaxis]
        + sigma[..., np.newaxis] * norm.ppf(eval_grid)[np.newaxis, ...]
    )
    return [Function(eval_grid, y) for y in ys]


def gen_sample_points_from_qfs(
    quantile_functions: list[Function], size: int = 100, seed: int = None,
):
    """Generate a sample of observation points for given distributions.

    Need to supply quantile functions.

    """
    which_quantiles = np.random.default_rng(seed).uniform(0, 1, size)
    return np.array([qf[which_quantiles] for qf in quantile_functions])


def gen_params_scenario_one(num_of_distr: int, seed: int = 28071995) -> tuple:
    """Generate parameters for the density samples and define appropriate grids.

    This is the first simulation scenario of Petersen & Müller (2016).

    """
    # Draw different sigmas
    log_sigmas = np.random.default_rng(seed=seed).uniform(-1.5, 1.5, num_of_distr)
    mus = np.zeros(num_of_distr)
    sigmas = np.exp(log_sigmas)

    return mus, sigmas


def gen_params_scenario_two(num_of_distr: int, seed: int = 28071995) -> tuple:
    """Generate parameters for the density samples and define appropriate grids.

    This is the second simulation scenario of Petersen & Müller (2016).

    """
    # Draw different mus
    mus = np.random.default_rng(seed=seed).uniform(-3, 3, num_of_distr)
    sigmas = np.ones(num_of_distr)

    return mus, sigmas


def gen_truncnorm_pdf_points(
    a: float,
    b: float,
    mu: float,
    sigma: float,
    sample_points_num: int = 100,
) -> tuple:
    """Generate sample points of truncated normal distributions characterized by the the
    mus and sigmas.

    This function is for

    """
    # For vectorized generation of sample points
    a = np.ones(len(mu)) * a
    b = np.ones(len(mu)) * b
    a = a[:, np.newaxis]
    b = b[:, np.newaxis]
    mu = mu[:, np.newaxis]
    sigma = sigma[:, np.newaxis]

    a_std = (a - mu) / sigma
    b_std = (b - mu) / sigma

    return truncnorm.rvs(
        a=a_std,
        b=b_std,
        loc=mu,
        scale=sigma,
        size=(len(a), sample_points_num),
    )


def make_estimated_pdf(
    sample_points: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    kern: str = "epanechnikov",
    grid_size: int = 10000,
    bandwidth: np.ndarray = 0.2,
    bias_corrected: bool = True,
) -> list[Function]:
    """Turn sample_points array into Function class objects with their corresponding
    grid.

    Needs a and b to be vectors of the same length as the number of rows of
    sample_points. The number of rows corresponds to the number of densities.

    """
    pdfs_x = np.linspace(a, b, grid_size).transpose()
    # Check if we're only dealing with one single density
    if bias_corrected:
        return boundary_corrected_density_estimator(
            x_vals=pdfs_x, sample_of_points=sample_points, h=bandwidth, kernel_type=kern,
        )
    else:
        return density_estimator(
            x_vals=pdfs_x, sample_of_points=sample_points, h=bandwidth, kernel_type=kern,
        )


# Truncated normal pdf
def make_truncnorm_pdf(
    a: np.ndarray = 0,
    b: np.ndarray = 1,
    mu: np.ndarray = 0,
    sigma: np.ndarray = 1,
    grid_size: int = 10000,
    warn_irregular_densities: bool = True,
) -> tuple:
    """Define truncated normal density function.

    To test: columns of x must align with mu and sigma.

    """
    pdf_x = np.linspace(a, b, grid_size)
    x = pdf_x[..., np.newaxis]  # to vectorize the input
    mu = np.array(mu)
    sigma = np.array(sigma)
    x_std = (x - mu) / sigma
    a_std = (a - mu) / sigma
    b_std = (b - mu) / sigma
    numerator = norm_pdf(x_std, 0, 1)
    denominator = _norm_cdf(b_std, 0, 1) - _norm_cdf(a_std, 0, 1)

    pdfs_y = numerator / denominator / sigma

    # Create a boolean mask for values outside the interval [a, b]
    mask = (x_std < a_std) | (x_std > b_std)

    # Set the PDF to zero for values of x outside the interval [a, b]
    pdfs_y[mask] = 0
    pdfs_y = pdfs_y.transpose()

    # Check whether each density integrates to 1
    pdfs_y = _check_and_normalize_density(
        x_vals=pdf_x,
        y_vals=pdfs_y,
        eps=1e-5,
        warn=warn_irregular_densities,
    )
    return [(pdf_x, pdf_y) for pdf_y in pdfs_y]


# Normal pdf
def norm_pdf(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Define normal density function.

    To test: columns of x must align with mu and sigma.

    """
    x = np.array(x)  # to vectorize the input
    mu = np.array(mu)
    sigma = np.array(sigma)
    return np.reciprocal(np.sqrt(2 * np.pi) * sigma) * np.exp(
        (-0.5) * ((x - mu) / sigma) ** 2,
    )


# Normal cdf
def _norm_cdf(
    x: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    grid_size: int = 1000,
) -> np.ndarray:
    """Compute the CDF of the normal distribution at a given point x.

    `x` can be a 2d array, with the number of rows corresponding to the number of
    densities n. Accordingly, `mu` and `sigma` can be vectors of length n.

    """
    minus_inf = -20  # Lower limit of integration (approximation of negative infinity)
    grid_to_integrate = np.linspace(minus_inf, x, grid_size).transpose()
    # Integrate the normal density function from a to b
    return riemann_sum_cumulative(
        grid_to_integrate,
        norm_pdf(grid_to_integrate, mu, sigma),
    )[1][..., -1]


def _check_and_normalize_density(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    eps: float = 1e-5,
    warn: bool = True,
) -> np.ndarray:
    """Checks whether y_vals is an array of valid densities, i.e., whether they
    integrate to one.

    Normalizes them to integrate to one. Positivity is not checked.

    """
    integrals = riemann_sum_cumulative(x_vals, y_vals, axis=-1)[1][..., -1]
    deviations_from_1 = abs(integrals - 1)
    if np.any(deviations_from_1 > eps):
        if warn:
            warnings.warn(
                f"Not all provided densities integrate to 1 with tolerance {eps}!"
                f"\n Max case of deviation is: {deviations_from_1.max()}"
                f"\n In position: {deviations_from_1.argmax()}"
                "\n Performing normalization...",
            )
        y_vals /= integrals[..., -1]
    return y_vals
