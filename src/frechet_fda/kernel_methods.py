"""This module contains functions for performing kernel density estimation."""

from functools import partial

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
    return riemann_sum_cumulative(
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
