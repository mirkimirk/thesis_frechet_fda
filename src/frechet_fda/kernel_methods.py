"""This module contains functions for performing kernel density estimation."""


import numpy as np

from frechet_fda.function_class import Function
from frechet_fda.function_tools import make_function_objects

kernels = {
    "epanechnikov": lambda u: 3
    / (4 * np.sqrt(5))
    * (1 - (u**2) / 5)
    * (np.abs(u) <= np.sqrt(5)),
    "uniform": lambda u: 0.5 * (np.abs(u) <= 1),
    "triangular": lambda u: (1 - np.abs(u)) * (np.abs(u) <= 1),
    "std_normal": lambda u: (1 / np.sqrt(2 * np.pi))
    * np.exp(-0.5 * u**2),  # p. 8 Li/Racine
}


def make_kernel_functions(x_vals: np.ndarray, kernel: str = "epanechnikov"):
    """Return kernel function as Function class object."""
    k = kernels[kernel]
    return [
        Function(kernel_x, kernel_y)
        for kernel_x, kernel_y in zip(x_vals, k(x_vals), strict=True)
    ]


def density_estimator(
    x_vals: np.ndarray,
    sample_of_points: np.ndarray,
    h,
    kernel_type="epanechnikov",
) -> list[Function]:
    """Kernel density estimator function.

    Assumes each row in `sample_of_points` is a density, and the columns represent the
    number of realizations. Each row in `x_vals` corresponds to the grid, on which
    density is to be estimated.

    """
    # Select kernel function
    k = kernels[kernel_type]
    # To make possibly scalar x_vals compatible with array operations
    pdfs_x = np.atleast_1d(x_vals)
    # Pre-allocate the result array for more speed
    result = np.zeros_like(pdfs_x)

    if sample_of_points.ndim > 1:
        for i, density in enumerate(sample_of_points):  # Looping over densities
            for point in density:  # Looping over samples for each density
                u = (pdfs_x[i] - point) / h
                result[i, :] += k(u)
        pdfs_y = result / (sample_of_points.shape[1] * h)
        list_of_densities = [
            (pdf_x, pdf_y) for pdf_x, pdf_y in zip(pdfs_x, pdfs_y, strict=True)
        ]
    else:
        # Add axes to make use of broadcasting rules and vectorization
        u = (pdfs_x[:, np.newaxis] - sample_of_points[np.newaxis, :]) / h
        result += np.sum(k(u), axis=1)
        pdfs_y = result / (len(sample_of_points) * h)
        list_of_densities = [
            (pdf_x, pdf_y) for pdf_x, pdf_y in zip(pdfs_x, pdfs_y, strict=True)
        ]
    return make_function_objects(list_of_densities)
