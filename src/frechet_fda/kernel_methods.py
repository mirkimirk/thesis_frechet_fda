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
    x_vals: np.ndarray,
    sample_of_points: np.ndarray,
    h,
    kernel_type="epanechnikov",
):
    """Kernel density estimator function. Assumes each row in `sample_of_points` is a
    density, and the columns represent the number of realizations. Each row in
    `x_vals` corresponds to the grid, on which density is to be estimated."""
    # Select kernel function
    k = kernels[kernel_type]
    # To make possibly scalar x_vals compatible with array operations
    x_vals = np.atleast_1d(x_vals)
    # Pre-allocate the result array for more speed
    result = np.zeros_like(x_vals)

    if sample_of_points.ndim > 1:
        n_densities, n_samples = sample_of_points.shape
        for i in range(n_densities):  # Looping over densities
            for j in range(n_samples):  # Looping over samples for each density
                u = (x_vals[i] - sample_of_points[i, j]) / h
                result[i, :] += k(u)
    else:
        n_samples = len(sample_of_points)
        # Add axes to make use of broadcasting rules and vectorization
        u = (x_vals[:, np.newaxis] - sample_of_points[np.newaxis, :]) / h
        result += np.sum(k(u), axis=1)
    return result / (n_samples * h)
