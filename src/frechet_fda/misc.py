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

    # Create a boolean mask for values outside the interval [a, b]
    mask = (x_std < a_std) | (x_std > b_std)

    # Set the PDF to zero for values of x outside the interval [a, b]
    result[mask] = 0
    result = result.transpose()

    # Check whether each density integrates to 1
    eps = 1e-5
    integrals = riemann_sum_arrays(np.linspace(a, b, len(x)), result, axis=-1)
    deviations_from_1 = abs(integrals - 1)
    if np.any(deviations_from_1 > eps):
        warnings.warn(
            f"Not all provided densities integrate to 1 with tolerance {eps}!"
            f"\n Max case of deviation is: {deviations_from_1.max()}"
            f"\n In position: {deviations_from_1.argmax()}"
            "\n Performing normalization...",
        )
        result /= integrals[..., np.newaxis]
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


def cdf_from_density(support_grid, density, axis, cumsum=True):
    """Calculate cdf values from discretized densities."""
    cdfs = riemann_sum_arrays(support_grid, density, axis=axis, cumsum=cumsum)
    # Check whether each density integrates to 1
    eps = 1e-3
    deviations_from_1 = abs(cdfs[..., -1] - 1)
    if np.any(deviations_from_1 > eps):
        warnings.warn(
            "Not all provided densities integrate to 1!"
            f"\n Max case of deviation is: {deviations_from_1.max()}"
            f"\n In position: {deviations_from_1.argmax()}"
            "\n Performing normalization...",
        )
        cdfs /= cdfs[..., -1, np.newaxis]
    return cdfs


def quantile_from_cdf(x_grid, cdf_values, prob_levels):
    """Compute discretized quantiles grid from discretized cdfs.

    x_grid and prob_levels need to be of same shape as cdf_values.

    """
    # Apply np.searchsorted along each row of cdf_values
    idx = np.apply_along_axis(np.searchsorted, 1, cdf_values, prob_levels, side="left")

    # Clip indices to ensure they are within bounds
    idx = np.clip(idx, 1, cdf_values.shape[-1] - 1)

    # Use advanced indexing to get the corresponding x_grid values
    row_idx = np.arange(x_grid.shape[0])[:, np.newaxis]

    return x_grid[row_idx, idx]


def quantile_from_density(dens, dsup, qsup=None):
    """Compute quantiles from densities."""
    if qsup is None:
        qsup = np.linspace(0, 1, dens.shape[-1])

    eps = 1e-3
    if not np.allclose([np.min(qsup), np.max(qsup)], [0, 1], atol=eps):
        print(
            "Problem with support of the quantile domain's boundaries - resetting to default.",
        )
        qsup = np.linspace(0, 1, dens.shape[-1])

    integral_dens = riemann_sum_arrays(dsup, dens, axis=-1, cumsum=True)
    deviations_from_1 = abs(integral_dens[..., -1] - 1)
    if np.any(deviations_from_1 > eps):
        warnings.warn(
            f"Not all provided densities integrate to 1 with tolerance {eps}!"
            f"\n Max case of deviation is: {deviations_from_1.max()}"
            f"\n In position: {deviations_from_1.argmax()} "
            "\n Performing normalization...",
        )
        dens = dens / integral_dens[..., -1, np.newaxis]

    qsuptmp = integral_dens

    qtmp = dsup
    ind = np.unique(qsuptmp, return_index=True, axis=-1)[1]
    qsuptmp = qsuptmp[..., ind]
    qtmp = qtmp[..., ind]

    q = np.zeros(dens.shape)
    for i in range(len(dens)):
        q[i] = np.interp(qsup, qsuptmp[i], qtmp)

    return q


def dens_from_qd(qds_discretized, qdsup=None, dsup=None):
    """Compute density from a quantile density function.

    'Inspired' from qd2dens in fdadensity package in R.

    """
    # Validate input
    eps = 1e-3
    boundaries = [np.min(qdsup), np.max(qdsup)]
    if not np.allclose(boundaries, [0, 1], atol=eps):
        msg = f"Please check the support of the QF domain's boundaries: {boundaries}"
        raise ValueError(msg)

    integral_qd = riemann_sum_arrays(qdsup, array=qds_discretized, axis=-1, cumsum=True)
    if not np.isclose(integral_qd[-1], np.ptp(dsup), atol=eps):
        msg = (
            "Quantile Density does not integrate to the range of the densities with "
            f"tolerance {eps}."
            f"\n Integral is: {integral_qd[...,-1]}"
            f"\n Range is: {np.ptp(dsup)}"
        )
        raise ValueError(msg)

    # Calculate new support grid
    dtemp = dsup[0] + integral_qd

    # Calculate density
    dens_temp = 1 / qds_discretized
    idx_unique = np.unique(dtemp, return_index=True, axis=-1)[1]
    dtemp = dtemp[..., idx_unique]
    dens_temp = dens_temp[..., idx_unique]
    dens = np.interp(dsup, dtemp, dens_temp)

    # Normalize the density
    dens /= riemann_sum_arrays(dsup, dens, axis=-1, cumsum=False)[..., np.newaxis]

    return dens


def qd_from_dens(dens, dsup=None, qdsup=None):
    """Compute quantile densities directly from densities.

    'Inspired' from dens2qd in fdadensity package in R.

    """
    # Validate input
    eps = 1e-3
    boundaries = [np.min(qdsup), np.max(qdsup)]
    if not np.allclose(boundaries, [0, 1], atol=eps):
        msg = f"Please check the support of the QF domain's boundaries: {boundaries}"
        raise ValueError(msg)

    integral_dens = riemann_sum_arrays(dsup, array=dens, axis=-1, cumsum=False)
    deviations_from_1 = abs(integral_dens - 1)
    if np.any(deviations_from_1 > eps):
        warnings.warn(
            f"Not all provided densities integrate to 1 with tolerance {eps}!"
            f"\n Max case of deviation is: {deviations_from_1.max()}"
            f"\n In position: {deviations_from_1.argmax()} "
            "\n Performing normalization...",
        )
        dens /= integral_dens[..., np.newaxis]

    qd = 1 / dens
    integral_qd = riemann_sum_arrays(qdsup, qd, axis=-1, cumsum=False)
    qd *= np.ptp(dsup) / integral_qd[..., np.newaxis]

    return qd


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


def riemann_sum_arrays(support_grid, array, axis=-1, cumsum=False):
    """Computes Riemann sum for given array, along the axis that contains the grid of
    values.
    """
    # Calculate the step size between consecutive grid points
    step_sizes = np.diff(support_grid)
    # Repeat last element so the output is not one element shorter. Should be approx.
    # ok
    step_sizes = np.append(step_sizes, step_sizes[..., -1][..., np.newaxis], axis=-1)

    # Compute the cumulative sum along the specified axis (i.e.,
    # the integral up to each grid point)
    if cumsum:
        result = np.cumsum(array * step_sizes, axis=axis)
    # Or just the integral
    else:
        result = np.sum(array * step_sizes, axis=axis)

    # Return result
    return result


def l2_norm(support_grid, array, axis=-1, cumsum=False):
    """Compute L2 norm of (approximate) function."""
    return np.sqrt(
        riemann_sum_arrays(
            support_grid=support_grid,
            array=array**2,
            axis=axis,
            cumsum=cumsum,
        ),
    )


def wasserstein_frechet_mean(qds_discretized, dsup, qdsup=None):
    """Compute Wasserstein-Fréchet mean from sample."""
    if qdsup is None:
        qdsup = np.linspace(0, 1, qds_discretized.shape[-1])
    mean_qdf = np.mean(qds_discretized, axis=0)
    integral = riemann_sum_arrays(qdsup, array=mean_qdf, axis=-1, cumsum=False)
    mean_qdf *= (dsup[-1] - dsup[0]) / integral
    return dens_from_qd(mean_qdf, qdsup, dsup)


def quantile_distance(quantile_1, quantile_2, support_grid, cumsum=False):
    """Compute Wasserstein / Quantile distance."""
    diff_squared = (quantile_1 - quantile_2) ** 2
    return riemann_sum_arrays(
        support_grid=support_grid,
        array=diff_squared,
        axis=-1,
        cumsum=cumsum,
    )
