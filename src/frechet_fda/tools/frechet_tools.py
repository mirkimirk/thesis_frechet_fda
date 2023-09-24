"""Contains methods for Fréchet regression in the context of distribution responses."""

import numpy as np
import quadprog

from frechet_fda.function_class import Function
from frechet_fda.tools.function_tools import (
    mean_func, quantile_distance
)

def _empirical_weight_function(
        x_to_predict : float,
        x_observed : np.ndarray
    ) -> np.ndarray:
    """Weight function to compute weighted Fréchet mean."""
    x_observed = np.atleast_2d(x_observed)
    means = np.mean(x_observed, axis=-1)
    cov_matrix = (
        x_observed @ x_observed.transpose() / x_observed.shape[-1]
    )
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    return 1 + (
        (x_observed - means[..., np.newaxis]).transpose()
        @ inv_cov_matrix
        @ (x_to_predict - means)
    )


def qf_tilde(
        x_to_predict : float,
        x_observed : np.ndarray,
        qfs_observed : list[Function]
    ) -> Function:
    """Estimator for quantile function."""
    return mean_func(
        _empirical_weight_function(x_to_predict, x_observed) * qfs_observed
    )


def solve_frechet_qp(
        xs_to_predict : np.ndarray,
        x_observed : np.ndarray,
        quantile_functions : list[Function]
    ) -> list[Function]:
    """Sets up quadratic programming problem and solves it.
    """
    estimates = []
    for x in xs_to_predict:
        # Estimate condtional qf, drop support where values become nan or inf
        estimated_qf = qf_tilde(x, x_observed, quantile_functions).drop_inf()
        constraints_check = estimated_qf.y[1:] - estimated_qf.y[:-1]
        if np.all(constraints_check > 0):
            # If estimator valid qf, it is the optimal solution
            estimates.append(estimated_qf)
        else:
            # Else, find closest vector to estimator that is a valid solution
            qp_a = estimated_qf.y
            grid_size = len(qp_a)
            qp_g = np.identity(grid_size)  # make sure P is symmetric
            qp_c = (
                np.eye(grid_size, grid_size - 1, k=-1)
                - np.eye(grid_size, grid_size - 1)
            )
            qp_b = np.zeros(grid_size - 1)
            solution = quadprog.solve_qp(qp_g, qp_a, qp_c, qp_b)[0]
            estimates.append(Function(estimated_qf.x, solution))
    return estimates


def ise_wasserstein(
        m_hat : list[Function],
        true_m : list[Function],
        x_to_predict : np.ndarray,
        already_qf : bool = False
    ) -> float:
    """Compute integrated squared error."""
    distances = [
        quantile_distance(hat, true, already_qf=already_qf)
        for hat, true in zip(m_hat, true_m, strict=True)
    ]
    return Function(x_to_predict, distances).integrate().y[-1]