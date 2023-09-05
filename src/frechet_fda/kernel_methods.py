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


def density_estimator(x, sample, h, kernel_type="epanechnikov"):
    """Kernel density estimator function."""
    # Select kernel function
    k = kernels[kernel_type]
    # To make possibly scalar-valued x compatible with vectorized operations
    x = np.atleast_1d(x)
    u = (x[:, np.newaxis] - sample) / h
    return 1 / (len(sample) * h) * np.sum(k(u), axis=1)
