"""This module contains functions needed for functional data analysis methods."""

import numpy as np
from scipy.sparse.linalg import eigsh

from frechet_fda.distribution_class import Distribution
from frechet_fda.distribution_tools import mean_func
from frechet_fda.numerics_helpers import riemann_sum_cumulative


def compute_centered_data(function_sample: list[Distribution]) -> list[Distribution]:
    """Compute mean function, centered data, and covariance matrix."""
    # Compute the mean function
    mean_function = mean_func(function_sample)

    # Center the data
    centered_data = [function - mean_function for function in function_sample]
    centered_data_same_support = [func.standardize_shape() for func in centered_data]

    return mean_function, centered_data_same_support


def compute_cov_function(centered_sample: list[Distribution]) -> np.ndarray:
    """Compute discretized covariance function."""
    # Create an empty list to store the y-values of each Distribution instance
    y_values_list = []

    # Loop through each Distribution instance and append its y-values to y_values_list
    for dist in centered_sample:
        y_values_list.append(dist.y)

    # Convert the list of y-values to a 2D NumPy array
    y_values_matrix = np.array(y_values_list)

    # Compute the covariance matrix
    cov_matrix = np.cov(y_values_matrix, rowvar=False)

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

    eigenfunctions = [
        Distribution(x_vals, eigenfunc) for eigenfunc in eigenfunctions_sorted
    ]

    # Scale each column of the eigenfunctions matrix by its respective L^2 norm
    eigenfunctions_scaled = [
        eigenfunction / eigenfunction.l2norm() for eigenfunction in eigenfunctions
    ]

    return eigenvalues_sorted, eigenfunctions_scaled


def compute_fpc_scores(
        x_vals : np.ndarray,
        centered_sample : list[Distribution],
        eigenfunctions_trunc : list[Distribution]
    ):
    """Computes factor loadings / FPC scores."""
    # Collect function values from Distribution objects
    y_values_densities = []
    y_values_eigfuncs = []
    # Loop through each Distribution instance and append its y-values to a list
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


def mode_of_variation(mean, eigval, eigfunc, alpha):
    """Compute kth mode of variation."""
    return mean + alpha * np.sqrt(eigval) * eigfunc
