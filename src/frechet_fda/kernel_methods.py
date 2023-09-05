"""This module contains functions for performing kernel density estimation."""


import numpy as np

kernels = {
    "epanechnikov": lambda u: 3
    / (4 * np.sqrt(5))
    * (1 - (u**2) / 5)
    * (np.abs(u) <= np.sqrt(5)),
    "uniform": lambda u: 0.5 * (np.abs(u) <= 1),
    "triangular": lambda u: (1 - np.abs(u)) * (np.abs(u) <= 1),
    "std_normal": lambda u: (1 - np.sqrt(2 * np.pi))
    * np.exp(-0.5 * u),  # p. 8 Li/Racine
}


def density_estimator(
    x: np.ndarray,
    sample_of_points: np.ndarray,
    h,
    kernel_type="epanechnikov",
):
    """Kernel density estimator function."""
    # Select kernel function
    k = kernels[kernel_type]
    n_densities, n_samples = sample_of_points.shape
    # To make possibly scalar x compatible with array operations
    x = np.atleast_1d(x)

    # Pre-allocate the result array for more speed
    result = np.zeros((n_densities, x.shape[0]))

    for i in range(n_densities):  # Looping over densities
        for j in range(n_samples):  # Looping over samples for each density
            u = (x - sample_of_points[i, j]) / h
            result[i, :] += k(u)
    return result / (n_samples * h)
