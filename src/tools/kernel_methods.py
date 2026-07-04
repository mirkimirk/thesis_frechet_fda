"""This module contains functions for performing kernel density estimation."""


import numpy as np

from function_class import Function
from tools.function_tools import make_function_objects
from tools.numerics_tools import riemann_sum, riemann_sum_cumulative

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


# This function is not used after all...
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
    h: np.ndarray,
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
    sample_of_points = np.asarray(sample_of_points)
    h = np.asarray(h)

    if sample_of_points.ndim > 1:
        if h.ndim == 0:
            u = (
                (pdfs_x[:, np.newaxis, :] - sample_of_points[:, :, np.newaxis])
                / h
            )
            pdfs_y = np.sum(k(u), axis=1) / (sample_of_points.shape[1] * h)
        elif h.shape[0] == sample_of_points.shape[1]:
            # Preserve the notebook's existing point-wise bandwidth convention.
            u = (
                (pdfs_x[:, np.newaxis, :] - sample_of_points[:, :, np.newaxis])
                / h[np.newaxis, :, np.newaxis]
            )
            pdfs_y = np.sum(k(u), axis=1) / (sample_of_points.shape[1] * h[-1])
        elif h.shape[0] == sample_of_points.shape[0]:
            u = (
                (pdfs_x[:, np.newaxis, :] - sample_of_points[:, :, np.newaxis])
                / h[:, np.newaxis, np.newaxis]
            )
            pdfs_y = np.sum(k(u), axis=1) / (
                sample_of_points.shape[1] * h[:, np.newaxis]
            )
        else:
            msg = "Bandwidth must be scalar, per sample point, or per density."
            raise ValueError(msg)
        list_of_densities = [
            (pdf_x, pdf_y) for pdf_x, pdf_y in zip(pdfs_x, pdfs_y, strict=True)
        ]
    else:
        # Add axes to make use of broadcasting rules and vectorization
        u = (pdfs_x[:, np.newaxis] - sample_of_points[np.newaxis, :]) / h
        pdfs_y = np.sum(k(u), axis=1) / (len(sample_of_points) * h)
        list_of_densities = [(pdfs_x, pdfs_y)]
    return make_function_objects(list_of_densities)


def boundary_corrected_density_estimator(
    x_vals: np.ndarray,
    sample_of_points: np.ndarray,
    h: float,
    kernel_type: str = "epanechnikov",
):
    """Calculate boundary corrected density estimator from Petersen & Müller 2016."""
    def standardize(r):
        numerator = r - x_vals[:, 0][:, np.newaxis]
        denominator = (x_vals[:, -1][:, np.newaxis] - x_vals[:, 0][:, np.newaxis])
        return numerator / denominator
    x_std = standardize(x_vals)
    sample_std = standardize(sample_of_points)
    h_std = h / (x_vals[:, -1] - x_vals[:, 0])
    k = kernels[kernel_type]

    w_xh = np.zeros_like(x_std)

    if sample_std.ndim > 1:
        for i in range(sample_std.shape[0]):
            w_xh[i] = _weight_function(x_vals=x_std[i], h=h_std[i], kernel=kernel_type)
        u = (
            (x_std[:, np.newaxis, :] - sample_std[:, :, np.newaxis])
            / h_std[:, np.newaxis, np.newaxis]
        )
        numerator = np.sum(k(u), axis=1) * w_xh
        denominator = riemann_sum(x_std, numerator)
    else:
        raise ValueError ("Not implemented for single densities yet!")
    kernel_y = (
        numerator / denominator[:, np.newaxis]
        / (x_vals[:, -1][:, np.newaxis] - x_vals[:, 0][:, np.newaxis])
    )
    return make_function_objects([(x, y) for x, y in zip(x_vals, kernel_y, strict=True)])


def _weight_function(x_vals: np.ndarray, h: float, kernel: str = "epanechnikov"):
    """Weight function for use in Petersen & Müller's 2016 estimator. Makes use
    of the symmetry of the kernel function when calculating integrals."""
    k = kernels[kernel]
    weight = np.zeros_like(x_vals)
    
    # Case for x_vals in [0, h)
    mask1 = (x_vals >= 0) & (x_vals < h)
    x1 = x_vals[mask1]
    integral1 = (
        riemann_sum(x1 / h, k(x1 / h))
        + riemann_sum_cumulative(x1 / h, k(x1 / h))[1]
    )
    weight[mask1] = np.reciprocal(integral1)
    
    # Case for x_vals in (1-h, 1]
    mask2 = (x_vals > 1 - h) & (x_vals <= 1)
    x2 = x_vals[mask2]
    integral2 = (
        2 * riemann_sum((x2 - 1) / h, k((x2 - 1) / h))
        - riemann_sum_cumulative((x2 - 1) / h, k((x2 - 1) / h))[1]
    )
    weight[mask2] = np.reciprocal(integral2)
    
    # Case for x_vals in [h, 1-h]
    mask3 = (x_vals >= h) & (x_vals <= 1 - h)
    weight[mask3] = weight[mask2][0]
    
    return weight
