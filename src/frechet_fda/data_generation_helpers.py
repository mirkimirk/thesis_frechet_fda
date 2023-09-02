"""This module contains code for generating data from a truncated normal distribution,
and functions that generate parameters according to the three scenarios outlined in
Petersen and Mueller (2016)'s simulation study."""

import warnings

import numpy as np
from frechet_fda.numerics_helpers import riemann_sum_cumulative


def gen_params_scenario_one(
    num_of_distr: int,
) -> tuple:
    """Generate parameters for the density samples and define appropriate grids."""
    # Draw different sigmas
    log_sigmas = np.random.default_rng(seed=28071995).uniform(-1.5, 1.5, num_of_distr)
    mus = np.zeros(num_of_distr)
    sigmas = np.exp(log_sigmas)

    return mus, sigmas


# Truncated normal pdf
def make_truncnorm_pdf(
    a: float = 0,
    b: float = 1,
    mu: np.ndarray = 0,
    sigma: np.ndarray = 1,
    grid_size: int = 1000
) -> tuple:
    """Define truncated normal density function.

    To test: columns of x must align with mu and sigma.

    """
    pdf_x = np.linspace(a, b, grid_size)
    x = pdf_x[..., np.newaxis]  # to vectorize the input
    mu = np.array(mu)
    sigma = np.array(sigma)
    x_std = (x - mu) / sigma
    a_std = (a - mu) / sigma
    b_std = (b - mu) / sigma
    numerator = _norm_pdf(x_std, 0, 1)
    denominator = _norm_cdf(b_std, 0, 1) - _norm_cdf(a_std, 0, 1)

    pdfs_y = numerator / denominator / sigma

    # Create a boolean mask for values outside the interval [a, b]
    mask = (x_std < a_std) | (x_std > b_std)

    # Set the PDF to zero for values of x outside the interval [a, b]
    pdfs_y[mask] = 0
    pdfs_y = pdfs_y.transpose()

    # Check whether each density integrates to 1
    eps = 1e-5
    integrals = riemann_sum_cumulative(
        np.linspace(a, b, len(x)), pdfs_y, axis=-1
    )[1][..., -1]
    deviations_from_1 = abs(integrals - 1)
    if np.any(deviations_from_1 > eps):
        warnings.warn(
            f"Not all provided densities integrate to 1 with tolerance {eps}!"
            f"\n Max case of deviation is: {deviations_from_1.max()}"
            f"\n In position: {deviations_from_1.argmax()}"
            "\n Performing normalization...",
        )
        pdfs_y /= integrals[..., -1]
    return pdf_x, pdfs_y




# Normal pdf
def _norm_pdf(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
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
def _norm_cdf(
        x: np.ndarray, mu: np.ndarray, sigma: np.ndarray, grid_size : int = 1000
    ) -> np.ndarray:
    """Compute the CDF of the normal distribution at a given point x.

    `x` can be a 2d array, with the number of rows corresponding to the number of
    densities n. Accordingly, `mu` and `sigma` can be vectors of length n.

    """
    minus_inf = -20  # Lower limit of integration (approximation of negative infinity)
    grid_to_integrate = np.linspace(minus_inf, x, grid_size).transpose()
    # Integrate the normal density function from a to b
    return riemann_sum_cumulative(
        grid_to_integrate, _norm_pdf(grid_to_integrate, mu, sigma),
    )[1][..., -1]
