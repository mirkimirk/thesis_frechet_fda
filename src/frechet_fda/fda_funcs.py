"""This module contains functions needed for functional data analysis methods."""

import numpy as np


from misc import (
    l2_norm,
    riemann_sum_arrays
)

def compute_moments(density_sample):
    """Compute mean function, centered data, and covariance matrix."""
    # Compute the mean function
    mean_function = np.mean(density_sample, axis=0)

    # Center the data
    centered_densities = density_sample - mean_function

    # Estimate the covariance function using a discrete approximation
    cov_matrix = np.cov(centered_densities, rowvar=False)

    return mean_function, centered_densities, cov_matrix


def compute_principal_components(cov_matrix, support_grid):
    """Compute functional principal components of a covariance function."""
    # Compute the eigenfunctions (principal components) of the covariance matrix
    eigenvalues, eigenfunctions = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvalues and eigenfunctions in decreasing order
    eigenvalues_sorted = eigenvalues[np.argsort(-eigenvalues)]
    eigenfunctions_sorted = eigenfunctions[:, np.argsort(-eigenvalues)]

    # Compute the L2 norm for each column (eigenvector) for rescaling to L2 norm
    l2_norms = l2_norm(
        support_grid=support_grid,
        array=eigenfunctions_sorted,
        axis=1,
        cumsum=False,
    )

    # Scale each column of the eigenfunctions matrix by its respective L^2 norm
    eigenfunctions_scaled = eigenfunctions_sorted / l2_norms

    return eigenvalues_sorted, eigenfunctions_scaled


def compute_fpc_scores(centered_densities, eigenfunctions, support_grid):
    """Computes factor loadings / FPC scores."""
    # Compute FPC scores / factor loadings
    products = np.einsum("ij,jk->ijk", centered_densities, eigenfunctions)
    return riemann_sum_arrays(
        support_grid=support_grid,
        array=products,
        axis=1,
        cumsum=False
    )


def mode_of_variation(mean_func, eigval, eigfunc, alpha):
    """Compute kth mode of variation."""
    if np.ndim(eigval) != 0:
        mean_func = mean_func[:, np.newaxis]
    return mean_func + alpha * np.sqrt(eigval) * eigfunc