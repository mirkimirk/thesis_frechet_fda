"""Contains various functions needed in different modules."""
import warnings
from functools import partial

import numpy as np

kernels = {
    "epanechnikov": lambda u: 3
    / (4 * np.sqrt(5))
    * (1 - (u**2) / 5)
    * (np.abs(u) <= np.sqrt(5)),
    "uniform": lambda u: 0.5 * (np.abs(u) <= 1),
    "triangular": lambda u: (1 - np.abs(u)) * (np.abs(u) <= 1),
}


# Normal density
def norm_pdf(x, mu, sigma):
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
def norm_cdf(x, mu, sigma):
    """Compute the CDF of the normal distribution at a given point x."""
    a = -10  # Lower limit of integration (approximation of negative infinity)
    b = x  # Upper limit of integration
    # Integrate the normal density function from a to b
    return riemann_sum(a, b, lambda y: norm_pdf(y, mu, sigma))


# Truncated normal pdf
def trunc_norm_pdf(x, mu, sigma, a, b):
    """Define truncated normal density function.

    To test: columns of x must align with mu and sigma.

    """
    x = np.array(x)  # to vectorize the input
    mu = np.array(mu)
    sigma = np.array(sigma)
    x_std = (x - mu) / sigma
    a_std = (a - mu) / sigma
    b_std = (b - mu) / sigma
    numerator = norm_pdf(x_std, 0, 1)
    denominator = norm_cdf(b_std, 0, 1) - norm_cdf(a_std, 0, 1)

    result = numerator / denominator / sigma

    # Set the PDF to zero for values of x outside the interval [a, b]
    result[(x < a) | (x > b)] = 0

    return result


def density_estimator(x, h, sample, kernel_type="epanechnikov"):
    """Kernel density estimator function."""
    # Select kernel function
    k = kernels[kernel_type]
    # To make possibly scalar-valued x compatible with vectorized operations
    x = np.atleast_1d(x)
    u = (x[:, np.newaxis] - sample) / h
    return 1 / (len(sample) * h) * np.sum(k(u), axis=1)


def cdf_estimator(
    x,
    h,
    sample,
    left_bound=-100,
    method="midpoint",
    step_size=None,
    kernel_type="epanechnikov",
):
    """Nonparametric cdf estimator (Li and Racine 2007, p.20)."""
    kd_estimator = partial(
        density_estimator,
        h=h,
        sample=sample,
        kernel_type=kernel_type,
    )
    return riemann_sum(
        a=left_bound,
        b=x,
        f=kd_estimator,
        method=method,
        step_size=step_size,
    )


def quantile_estimator(
    prob_levels,
    h,
    sample,
    x_grid,
    left_bound=-100,
    method="midpoint",
    step_size=None,
    kernel_type="epanechnikov",
):
    """Estimator of quantiles."""
    # Compute the CDF values for the x_grid
    prob_levels = np.atleast_1d(prob_levels)
    cdf_values = cdf_estimator(
        x=x_grid,
        h=h,
        sample=sample,
        left_bound=left_bound,
        method=method,
        step_size=step_size,
        kernel_type=kernel_type,
    )

    # Find the quantiles for the desired probability levels
    idx = np.searchsorted(cdf_values, prob_levels)
    quantiles = x_grid[idx - 1]

    return np.array(quantiles)


def cdf_from_density(support_grid, density, axis):
    """Calculate cdf values from discretized densities."""
    cdfs = riemann_sum_arrays(support_grid, density, axis=axis)
    if np.any(abs(cdfs[..., -1] - 1) > 1e-2):
        warnings.warn(
            "Not all provided densities integrate to 1!"
            f"\n Min case is: {cdfs[..., -1].min()} "
            f"\n Max case is: {cdfs[..., -1].max()} "
            "\n Performing normalization...",
        )
    cdfs /= cdfs[..., -1, np.newaxis]
    return cdfs


def quantiles_from_cdf(x_grid, cdf_values, prob_levels):
    """Compute discretized quantiles grid from discretized cdfs.

    x_grid and prob_levels need to be of same shape as cdf_values.

    """
    # Find where the CDF values are less than the probability levels
    condition = cdf_values < prob_levels

    # Find the indices of the first True value along the last axis
    idx = np.argmax(condition, axis=-1)

    # Clip the indices to ensure they are within bounds
    idx = np.clip(idx, 1, len(x_grid) - 1)

    # Retrieve the corresponding quantiles from x_grid
    quantiles = x_grid[idx]
    return quantiles


def density_from_qd(qd, dSup, qdSup=None):
    """Compute density from a quantile density function.

    'Inspired' from qd2dens in fdadensity package in R.

    """
    if qdSup is None:
        qdSup = np.linspace(0, 1, len(qd))
    dtemp = dSup[0] + riemann_sum_arrays(qdSup[0], qdSup, qd, axis=0)

    dens_temp = 1 / qd
    ind = np.unique(dtemp, return_index=True)[1]
    dtemp = np.atleast_1d(dtemp)[ind]
    dens_temp = dens_temp[~ind]
    dens = np.interp(dSup, dtemp, dens_temp)
    dens /= riemann_sum_arrays(dSup[0], dSup[-1], dens, axis=0)

    return dens


def riemann_sum(a, b, f, method="midpoint", step_size=None):
    """Approximate integral from scalar a to any value in vector b."""
    # For calculating cdf values when grid b is supplied.
    b = np.atleast_1d(b)
    max_b = np.max(b)
    if step_size is None:
        step_size = (max_b - a) / 1000
    m = int((max_b - a) / step_size)
    if method == "left":
        grid = np.linspace(a, max_b - step_size, m)
    elif method == "right":
        grid = np.linspace(a + step_size, max_b, m)
    elif method == "midpoint":
        grid = np.linspace(a, max_b, m + 1)
        grid = (grid[1:] + grid[:-1]) / 2
    else:
        msg = "Must specify either left, right, or midpoint Riemann sum!"
        raise ValueError(msg)
    values = f(grid) * step_size
    cdf_values = np.cumsum(values)
    return np.interp(b, grid, cdf_values)


# def riemann_sum_arrays(left_bound, right_bound, array, axis):
#     """"Computes riemann sum for given array, along the axis that contains the grid of
#     values.
#     """

#     # Compute the Riemann sum along the axis of grid values using vectorized computation


def riemann_sum_arrays(support_grid, array, axis):
    """Computes Riemann sum for given array, along the axis that contains the grid of
    values.
    """
    # Calculate the step size between consecutive grid points
    step_size = (support_grid[-1] - support_grid[0]) / (len(support_grid) - 1)

    # Compute the cumulative sum along the specified axis (i.e., the integral up to each grid point)
    cumulative_sums = np.cumsum(array, axis=axis) * step_size

    # Return the cumulative sums, which represent the CDF at each grid point
    return cumulative_sums


def l2_norm(left_bound_support, right_bound_support, array, axis):
    """Compute L2 norm of (approximate) function."""
    return np.sqrt(
        riemann_sum_arrays(
            left_bound=left_bound_support,
            right_bound=right_bound_support,
            array=array**2,
            axis=axis,
        ),
    )


def quantile_distance(quantile_1, quantile_2):
    """Compute Wasserstein / Quantile distance."""
    diff_squared = (quantile_1 - quantile_2) ** 2
    return riemann_sum_arrays(left_bound=0, right_bound=1, array=diff_squared, axis=0)
