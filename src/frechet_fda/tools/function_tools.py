"""This module contains tools to use with the distribution class."""

import numpy as np

from frechet_fda.function_class import Function


def make_function_objects(xy_tuples: list[tuple]) -> list[Function]:
    """Takes list of tuples of a support range and the associated function values, and
    gives a Function object.
    """
    return [Function(*xy_tuple) for xy_tuple in xy_tuples]


def pdf_to_qdf(pdf: Function, save_support_start : bool = False) -> Function:
    """Directly convert a pdf to a qdf using inverse function rule on qf."""
    quantile_func = pdf.integrate().invert()
    if save_support_start:
        return (1 / pdf.compose(quantile_func), quantile_func.y[0])
    else:
        return 1 / pdf.compose(quantile_func)


def qdf_to_pdf(
        qdf: Function, start_val : float = 0, center_on_zero : bool = False
    ) -> Function:
    """Directly convert a qdf to a pdf using inverse function rule on cdf."""
    if center_on_zero:
        cdf = qdf.integrate().vcenter().invert()
    else:
        cdf = (qdf.integrate() + start_val).invert()
    return 1 / qdf.compose(cdf)


def get_optimal_range(funcs: list[Function], delta: float = 1e-3) -> np.ndarray:
    """Get narrower support if density values are too small (smaller than delta).

    This is used so the qdfs dont get astronomically large at the boundaries and destroy
    numerical methods.
    
    Note: The method here assumes that the functions do have a compact
    support (even if it is narrower than the initial support). So if there is a point x1
    where func.y > delta is true, another point x2 > x1 where it is not, and then
    another x3 > x2 where it is true again, then x2 is included in the new range
    although it does not fullfill the condition.

    """
    new_ranges = np.zeros((len(funcs), 2))
    for i, func in enumerate(funcs):
        support_to_keep = func.x[func.y > delta]
        new_ranges[i] = (support_to_keep[0], support_to_keep[-1])
    return new_ranges


def mean_func(funcs: list[Function]) -> Function:
    """Compute the mean of a list of functions (instances of Function class)."""
    num_funcs = len(funcs)
    agg_func = funcs[0] / num_funcs
    for i in range(1, num_funcs):
        agg_func += funcs[i] / num_funcs
    return agg_func


def log_qd_transform(
        densities_sample: list[Function],
        different_supports : bool = False
    ) -> list[Function]:
    """Perfrom log quantile density transformation on a density sample."""
    qdfs = [
        pdf_to_qdf(density.drop_inf(), different_supports)
        for density in densities_sample
    ]
    if different_supports:
        qdfs = np.array(qdfs)
        lqdfs = [qdf.log() for qdf in qdfs[:, 0]]
        return np.array((lqdfs, qdfs[:, 1])).transpose()
    else:
        return [qdf.log() for qdf in qdfs]


def inverse_log_qd_transform(
    transformed_funcs : list[Function],
    start_of_support : list[float] = None
) -> list[Function]:
    """Transform back into density space."""
    natural_qfs = [func.exp().integrate() for func in transformed_funcs]
    if start_of_support is not None:
        natural_qfs += start_of_support
        cdfs = [qf.invert() for qf in natural_qfs]
    else:
        cdfs = [qf.vcenter().invert() for qf in natural_qfs]
    exponents = [
        -func.compose(cdf) for func, cdf in zip(transformed_funcs, cdfs, strict=True)
    ]
    return [exponent.exp() for exponent in exponents]


def inverse_log_qd_transform_corrected(
    transformed_funcs : list[Function],
    left_bound : float,
    right_bound : float,
    eps : float = 1e-3
) -> list[Function]:
    """Invert the log quantile density transform to get back into density space.
    
    Not used, I must understand something wrong about the implementation with a
    different support."""
    # First compute quantile function via natural inverse
    natural_qfs = [func.exp().integrate() for func in transformed_funcs]
    # Compute correction factors to normalize quantiles
    thetas = []
    corrected_qfs = []
    for qf in natural_qfs:
        if qf.y[-1] > (right_bound - left_bound) + eps:
            theta = qf.y[-1] / (right_bound - left_bound)
            thetas.append(theta)
            corrected_qfs.append((qf / theta) + left_bound)
        else:
            theta = qf.y[-1] / (qf.y[-1] - qf.y[0])
            thetas.append(theta)
            corrected_qfs.append((qf / theta).vcenter())
    cdfs = [qf.invert() for qf in corrected_qfs]
    exponents = [
        func.compose(cdf) for func, cdf in zip(transformed_funcs, cdfs, strict=True)
    ]
    inverses = [
        theta / exponent.exp()
        for theta, exponent in zip(thetas, exponents, strict=True)
    ]

    return inverses


def frechet_mean(
        density_sample: list[Function], centered_on_zero : bool = False
    ) -> Function:
    """Compute Fréchet mean of a given sample of densities.
    
    By default, estimates new support by taking the mean of the left bounds of the
    sample of densities. If centered_on_zero specified, then resulting mean density is
    transformed to be centered around zero.
    """
    qdfs_and_start_vals = np.array(
        [pdf_to_qdf(density, True) for density in density_sample]
    )
    qdfs = qdfs_and_start_vals[:, 0]
    start_vals = qdfs_and_start_vals[:, 1]
    mean_qdf = mean_func(qdfs)
    mean_start_val = np.mean(start_vals) # estimated start of the support of mean pdf
    return qdf_to_pdf(mean_qdf, mean_start_val, centered_on_zero)


def quantile_distance(
        distr1: Function, distr2: Function, already_qf : bool = False
    ) -> float:
    """Compute Wasserstein / Quantile distance for two given pdfs or qfs."""
    if already_qf:
        qf1, qf2 = distr1, distr2
    else:
        qf1 = distr1.integrate().invert()
        qf2 = distr2.integrate().invert()
    diff_squared = (qf1 - qf2) ** 2
    return diff_squared.integrate().y[-1]
