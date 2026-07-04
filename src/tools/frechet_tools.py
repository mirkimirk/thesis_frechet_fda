"""Contains methods for Fréchet regression in the context of distribution responses."""

import numpy as np
import quadprog
from function_class import Function
from tools.function_tools import (
    function_values_on_common_grid,
    quantile_distance,
)
from tools.numerics_tools import riemann_sum


def _empirical_weight_function(
    x_to_predict: float, x_observed: np.ndarray,
) -> np.ndarray:
    """Weight function to compute weighted Fréchet mean."""
    return _empirical_weight_matrix(np.asarray([x_to_predict]), x_observed)[0]


def _empirical_weight_matrix(
    xs_to_predict: np.ndarray,
    x_observed: np.ndarray,
) -> np.ndarray:
    """Compute empirical Fréchet regression weights for all prediction points."""
    x_observed = np.atleast_2d(x_observed)
    xs_to_predict = np.asarray(xs_to_predict)
    if xs_to_predict.ndim == 0:
        xs_to_predict = xs_to_predict.reshape(1, 1)
    elif x_observed.shape[0] == 1:
        xs_to_predict = xs_to_predict.reshape(-1, 1)
    else:
        xs_to_predict = np.atleast_2d(xs_to_predict)
        if xs_to_predict.shape[0] == x_observed.shape[0]:
            xs_to_predict = xs_to_predict.transpose()

    means = np.mean(x_observed, axis=-1)
    cov_matrix = x_observed @ x_observed.transpose() / x_observed.shape[-1]
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    centered_observed = (x_observed - means[..., np.newaxis]).transpose()
    centered_predict = xs_to_predict - means
    return 1 + centered_predict @ inv_cov_matrix @ centered_observed.transpose()


def qf_tilde(
    x_to_predict: float, x_observed: np.ndarray, qfs_observed: list[Function],
) -> Function:
    """Estimator for quantile function."""
    weights = _empirical_weight_function(x_to_predict, x_observed)
    qf_x, qf_y = function_values_on_common_grid(qfs_observed)
    return Function(qf_x, weights @ qf_y / len(qfs_observed))


def solve_frechet_qp(
    xs_to_predict: np.ndarray,
    x_observed: np.ndarray,
    quantile_functions: list[Function],
) -> list[Function]:
    """Sets up quadratic programming problem and solves it."""
    weights = _empirical_weight_matrix(xs_to_predict, x_observed)
    qf_x, qf_y = function_values_on_common_grid(quantile_functions)
    estimated_qfs = weights @ qf_y / len(quantile_functions)

    estimates = []
    for estimated_y in estimated_qfs:
        # Estimate condtional qf, drop support where values become nan or inf
        estimated_qf = Function(qf_x, estimated_y).drop_inf()
        constraints_check = estimated_qf.y[1:] - estimated_qf.y[:-1]
        if np.all(constraints_check > 0):
            # If estimator valid qf, it is the optimal solution
            estimates.append(estimated_qf)
        else:
            # Else, find closest vector to estimator that is a valid solution
            qp_a = estimated_qf.y
            grid_size = len(qp_a)
            qp_g = np.identity(grid_size)
            qp_c = np.eye(grid_size, grid_size - 1, k=-1) - np.eye(
                grid_size, grid_size - 1,
            )
            qp_b = np.zeros(grid_size - 1)
            solution = quadprog.solve_qp(qp_g, qp_a, qp_c, qp_b)[0]
            estimates.append(Function(estimated_qf.x, solution))
    return estimates


def ise_wasserstein(
    m_hat: list[Function],
    true_m: list[Function],
    x_to_predict: np.ndarray,
    already_qf: bool = False,
) -> float:
    """Compute integrated squared error."""
    if already_qf:
        m_hat = list(m_hat)
        true_m = list(true_m)
        first_grid = m_hat[0].x
        same_grid = all(
            len(hat.x) == len(first_grid)
            and len(true.x) == len(first_grid)
            and np.array_equal(hat.x, first_grid)
            and np.array_equal(true.x, first_grid)
            for hat, true in zip(m_hat, true_m, strict=True)
        )
        if same_grid:
            hat_y = np.vstack([hat.y for hat in m_hat])
            true_y = np.vstack([true.y for true in true_m])
            distances = riemann_sum(
                first_grid,
                (hat_y - true_y) ** 2,
                method="midpoint",
            )
            return Function(x_to_predict, distances).integrate(method="midpoint").y[-1]

    distances = [
        quantile_distance(hat, true, already_qf=already_qf)
        for hat, true in zip(m_hat, true_m, strict=True)
    ]
    return Function(x_to_predict, distances).integrate(method="midpoint").y[-1]
