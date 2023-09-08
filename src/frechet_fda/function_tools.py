"""This module contains tools to use with the distribution class."""

import numpy as np

from frechet_fda.function_class import Function


def make_function_objects(xy_tuples: list[tuple]) -> list[Function]:
    """Takes list of tuples of a support range and the associated function values, and
    gives a Function object.
    """
    return [Function(*xy_tuple) for xy_tuple in xy_tuples]


def pdf_to_qdf(pdf: Function) -> Function:
    """Directly convert a pdf to a qdf using inverse function rule on qf."""
    quantile_func = pdf.integrate().invert()
    return 1 / pdf.compose(quantile_func)


def qdf_to_pdf(qdf: Function) -> Function:
    """Directly convert a qdf to a pdf using inverse function rule on cdf."""
    cdf = qdf.integrate().vcenter().invert()
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


def log_qd_transform(densities_sample: list[Function]) -> list[Function]:
    """Perfrom log quantile density transformation on a density sample."""
    qdfs = [pdf_to_qdf(density) for density in densities_sample]
    return [qdf.log() for qdf in qdfs]


def inverse_log_qd_transform(
    transformed_funcs: list[Function],
) -> list[Function]:
    """Transform back into density space."""
    natural_qfs = [func.exp().integrate().vcenter() for func in transformed_funcs]
    cdfs = [qf.invert() for qf in natural_qfs]
    exponents = [
        -func.compose(cdf) for func, cdf in zip(transformed_funcs, cdfs, strict=True)
    ]
    return [exponent.exp() for exponent in exponents]


def inverse_log_qd_transform_corrected(
    transformed_funcs: list[Function],
) -> list[Function]:
    """Invert the log quantile density transform to get back into density space."""
    # First compute quantile function via natural inverse
    natural_qfs = [func.exp().integrate().vcenter() for func in transformed_funcs]
    # Compute correction factors to normalize quantiles
    thetas = [qf.y[-1] for qf in natural_qfs]
    corrected_qfs = [qf / theta for qf, theta in zip(natural_qfs, thetas, strict=True)]
    cdfs = [qf.invert() for qf in corrected_qfs]
    exponents = [
        func.compose(cdf) for func, cdf in zip(transformed_funcs, cdfs, strict=True)
    ]
    inverses = [
        theta / exponent.exp()
        for theta, exponent in zip(thetas, exponents, strict=True)
    ]

    return inverses


def frechet_mean(density_sample: list[Function]) -> Function:
    """Compute Fréchet mean of a given sample of densities."""
    qdfs = [pdf_to_qdf(density) for density in density_sample]
    mean_qdf = mean_func(qdfs)
    return qdf_to_pdf(mean_qdf)


def quantile_distance(pdf1: Function, pdf2: Function) -> float:
    """Compute Wasserstein / Quantile distance."""
    diff_squared = (pdf1 - pdf2) ** 2
    return diff_squared.integrate().y[-1]
