"""This module contains functions needed for functional data analysis methods."""

import numpy as np
from scipy.sparse.linalg import eigsh

from frechet_fda.function_class import Function
from frechet_fda.tools.function_tools import (
    inverse_log_qd_transform,
    mean_func,
    quantile_distance,
)
from frechet_fda.tools.numerics_tools import riemann_sum_cumulative


def compute_mean_and_centered_data(function_sample: list[Function]) -> list[Function]:
    """Compute mean function, centered data, and covariance matrix."""
    # Compute the mean function
    mean_function = mean_func(function_sample)

    # Center the data
    centered_data = [function - mean_function for function in function_sample]
    centered_data_same_support = [func.set_grid_size() for func in centered_data]

    return mean_function, centered_data_same_support


def compute_cov_function(centered_sample: list[Function]) -> np.ndarray:
    """Compute discretized covariance function."""
    # Create an empty list to store the y-values of each Function instance
    y_values_list = []

    # Loop through each Function instance and append its y-values to y_values_list
    for dist in centered_sample:
        y_values_list.append(dist.y)

    # Convert the list of y-values to a 2D NumPy array
    y_values_matrix = np.array(y_values_list)

    # Compute the covariance matrix
    cov_matrix = y_values_matrix.transpose() @ y_values_matrix
    cov_matrix /= len(centered_sample)

    return cov_matrix


def compute_principal_components(
        x_vals: np.ndarray,
        cov_matrix: np.ndarray,
        k: int = 5,
    ) -> tuple:
    """Compute functional principal components of a covariance function."""
    # Compute the eigenfunctions (principal components) of the covariance matrix
    eigenvalues, eigenfunctions = eigsh(cov_matrix, k=k, which="LM")
    eigenfunctions = eigenfunctions.transpose()

    # Sort eigenvalues and eigenfunctions in decreasing order
    eigenvalues_sorted = eigenvalues[np.argsort(-eigenvalues)]
    eigenfunctions_sorted = eigenfunctions[np.argsort(-eigenvalues)]

    # Create Function objects, have them evaluated at the same points as the
    # centered_sample
    eigenfunctions = [
        Function(x_vals, eigenfunc).set_grid_size(grid_size=len(x_vals))
        for eigenfunc in eigenfunctions_sorted
    ]

    # Scale each eigenfunction by its respective L^2 norm
    eigenfunctions_scaled = [
        eigenfunction / eigenfunction.l2norm() for eigenfunction in eigenfunctions
    ]
    eigenvalues_scaled = eigenvalues_sorted / len(cov_matrix)

    return eigenvalues_scaled, eigenfunctions_scaled


def compute_fpc_scores(
        x_vals: np.ndarray,
        centered_sample: list[Function],
        eigenfunctions_trunc: list[Function],
    ):
    """Computes factor loadings / FPC scores."""
    # Collect function values from Function objects
    y_values_densities = []
    y_values_eigfuncs = []
    # Loop through each Function instance and append its y-values to a list
    for centered_func in centered_sample:
        y_values_densities.append(centered_func.y)
    for eigfunc in eigenfunctions_trunc:
        y_values_eigfuncs.append(eigfunc.y)
    # Convert the list of y-values to a 2D NumPy array
    y_values_densities_arr = np.array(y_values_densities)
    y_values_eigfuncs_arr = np.array(y_values_eigfuncs)
    # Compute FPC scores / factor loadings
    products = np.einsum("ij,kj->ikj", y_values_densities_arr, y_values_eigfuncs_arr)
    return riemann_sum_cumulative(x_vals=x_vals, y_vals=products)[1][..., -1]


def gen_qdtransformation_pcs(log_qdfs: list[Function], k: int = 5):
    """Perform FPCA on transformed densities."""
    mean, centered_data = compute_mean_and_centered_data(log_qdfs)
    covariance_function = compute_cov_function(centered_data)
    eigenvalues, eigenfunctions = compute_principal_components(
        centered_data[0].x,
        covariance_function,
        k=k,
    )
    fpc_scores = compute_fpc_scores(centered_data[0].x, centered_data, eigenfunctions)
    return mean, eigenvalues, eigenfunctions, fpc_scores


def mode_of_variation(mean: Function, eigval, eigfunc, alpha):
    """Compute kth mode of variation."""
    return mean + alpha * np.sqrt(eigval) * eigfunc


def karhunen_loeve(
        mean_function: Function,
        eigenfunctions: list[Function],
        fpc_scores: np.ndarray,
        K: int = 3,
    ) -> list[Function]:
    """Get truncated Karhunen-Loève representation."""
    truncated_reps = []
    for fpcs in fpc_scores:
        aggr = mean_function
        for k in range(K):
            aggr += eigenfunctions[k] * fpcs[k]
        truncated_reps.append(aggr)
    return truncated_reps


def total_frechet_variance(
        fmean: Function,
        densities_sample: list[Function],
    ) -> float:
    """Computes total frechet variance."""
    distances = []
    for density in densities_sample:
        distances.append(quantile_distance(density, fmean))
    return np.mean(distances)


def k_frechet_variance(
        total_var: float,
        densities_sample: list[Function],
        truncated_reps: list[Function],
    ) -> float:
    """Compute variance explained by truncated representation."""
    distances = []
    for density, trunc in zip(densities_sample, truncated_reps, strict=True):
        distances.append(quantile_distance(density, trunc))
    mean_dist = np.mean(distances)
    return total_var - mean_dist


def fve(
        total_var: Function,
        densities_sample: list[Function],
        truncated_representations: list[Function],
    ):
    """Compute Fréchet fraction of variance explained."""
    var_explained = k_frechet_variance(
        total_var,
        densities_sample,
        truncated_representations,
    )
    return var_explained / total_var, total_var, var_explained


def k_optimal(
        p: float,
        total_variance: float,
        densities_sample: list[Function],
        mean_function: Function,
        eigenfunctions: list[Function],
        fpc_scores: np.ndarray,
        save_specific_fve : int = None
    ) -> int:
    """Compute minimum number of components to include to reach ratio p of variance
    explained.

    ToDo: Simplify this function.

    save_specific_fve: Whether you want to save the fraction of variance explained for
    a particular truncation value k. If so, specify how many pcs to incorporate.
    """
    fv = 0
    k_opt = 0
    while fv < p:
        k_opt += 1
        trunc_reps_transforms = karhunen_loeve(
            mean_function,
            eigenfunctions,
            fpc_scores,
            K=k_opt,
        )
        trunc_reps = inverse_log_qd_transform(trunc_reps_transforms)
        fv = fve(total_variance, densities_sample, trunc_reps)[0]
        if (save_specific_fve is not None) & (k_opt == save_specific_fve):
            fv_of_interest = fv
    if (save_specific_fve is None):
        return k_opt, fv, trunc_reps
    else:
        return k_opt, fv, trunc_reps, fv_of_interest