"""This module contains tools to use with the distribution class."""

import numpy as np

from frechet_fda.distribution_class import Distribution


def make_distribution_objects(xy_tuples: list[tuple]) -> list[Distribution]:
    """Takes an x range and possibly multiple densitites defined on it in the y_array.

    Assumes that rows of

    """
    return [Distribution(*xy_tuple) for xy_tuple in xy_tuples]


def pdf_to_qdf(pdf: Distribution) -> Distribution:
    """Directly convert a pdf to a qdf using inverse function rule on qf."""
    quantile_func = pdf.integrate().invert()
    return 1 / pdf.compose(quantile_func)


def qdf_to_pdf(qdf: Distribution) -> Distribution:
    """Directly convert a qdf to a pdf using inverse function rule on cdf."""
    cdf = qdf.integrate().vcenter().invert()
    return 1 / qdf.compose(cdf)


def get_optimal_range(funcs: list[Distribution], delta: float = 1e-3):
    """Get narrower support if density values are too small (smaller than delta)."""
    new_ranges = np.zeros((len(funcs), 2))
    for i, func in enumerate(funcs):
        support_to_keep = func.x[func.y > delta]
        new_ranges[i] = (support_to_keep[0], support_to_keep[-1])
    return new_ranges


def mean_func(funcs: list[Distribution]) -> Distribution:
    """Compute the mean of a list of functions (instances of Distribution class)."""
    num_funcs = len(funcs)
    agg_func = funcs[0] / num_funcs
    for i in range(1, num_funcs):
        agg_func += funcs[i] / num_funcs
    return agg_func
