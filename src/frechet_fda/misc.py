"""Contains various functions needed in different modules."""
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
def norm_density(x, mu, sigma):
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
def norm_cdf(x, mu, sigma, m):
    """Compute the CDF of the normal distribution at a given point x."""
    a = -10  # Lower limit of integration (approximation of negative infinity)
    b = x  # Upper limit of integration
    # Integrate the normal density function from a to b
    return riemann_sum(a, b, m, lambda y: norm_density(y, mu, sigma))


def kernel_estimator(x, h, sample, kernel_type="epanechnikov"):
    """Kernel density estimator function."""
    # Select kernel function
    k = kernels[kernel_type]
    # To make possibly scalar-valued x compatible with vectorized operations
    x = np.atleast_1d(x)
    u = (x[:, np.newaxis] - sample) / h
    return 1 / (len(sample) * h) * np.sum(k(u), axis=1)


def cdf_estimator(
    x,
    f,
    h,
    sample,
    left_bound=-100,
    method="midpoint",
    step_size=None,
    kernel_type="epanechnikov",
):
    """Nonparametric cdf estimator (Li and Racine 2007, p. 20)."""
    kd_estimator = partial(f, h=h, sample=sample, kernel_type=kernel_type)
    return riemann_sum(
        a=left_bound,
        b=x,
        f=kd_estimator,
        method=method,
        step_size=step_size,
    )


def riemann_sum(a, b, f, method="midpoint", step_size=None):
    """Approximate integral from to scalar a to any value in vector b."""
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


def riemann_sum_arrays(left_bound, right_bound, array, axis):
    """"Computes riemann sum for given array, along the axis that contains the grid of
    values.
    """
    m = array.shape[axis]  # Number of points along the axis of grid values
    step_size = (right_bound - left_bound) / m

    # Compute the Riemann sum along the axis of grid values using vectorized computation
    return np.sum(array, axis=axis) * step_size


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

