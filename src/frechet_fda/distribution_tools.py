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


def get_optimal_range(funcs: list[Distribution], delta: float = 1e-3) -> np.ndarray:
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


def log_qd_transform(densities_sample : list[Distribution]) -> list[Distribution]:
    """Perfrom log quantile density transformation on a density sample."""
    qdfs = [pdf_to_qdf(density) for density in densities_sample]
    return [qdf.log() for qdf in qdfs]


def inverse_log_qd_transform(
        transformed_funcs : list[Distribution]
    ) -> list[Distribution]:
    natural_qfs = [func.exp().integrate().vcenter() for func in transformed_funcs]
    cdfs = [qf.invert() for qf in natural_qfs]
    exponents = [-func.compose(cdf) for func, cdf in zip(transformed_funcs, cdfs)]
    return [exponent.exp() for exponent in exponents]


def inverse_log_qd_transform_corrected(
        transformed_funcs : list[Distribution]
    ) -> list[Distribution]:
    """Invert the log quantile density transform to get back into density space."""
    # First compute quantile function via natural inverse
    natural_qfs = [func.exp().integrate().vcenter() for func in transformed_funcs]
    # Compute correction factors to normalize quantiles
    thetas = [qf.y[-1] for qf in natural_qfs]
    corrected_qfs = [
        qf / theta for qf, theta in zip(natural_qfs, thetas)
    ]
    cdfs = [qf.invert() for qf in corrected_qfs]
    exponents = [func.compose(cdf) for func, cdf in zip(transformed_funcs, cdfs)]
    inverses = [theta / exponent.exp() for theta, exponent in zip(thetas, exponents)]

    return inverses


def frechet_mean(density_sample : list[Distribution]) -> Distribution:
    """Compute Fréchet mean of a given sample of densities."""
    qdfs = [pdf_to_qdf(density) for density in density_sample]
    mean_qdf = mean_func(qdfs)
    return qdf_to_pdf(mean_qdf)


def quantile_distance(pdf1 : Distribution, pdf2 : Distribution) -> float:
    """Compute Wasserstein / Quantile distance."""
    diff_squared = (pdf1 - pdf2) ** 2
    return diff_squared.integrate().y[-1]


def total_frechet_variance(
        fmean : Distribution, densities_sample : list[Distribution]
    ) -> float:
    """Computes total frechet variance."""
    distances = []
    for density in densities_sample:
        distances.append(quantile_distance(density, fmean) ** 2)
    return np.mean(distances)


def k_frechet_variance(total_var, densities_sample, truncated_reps):
    distances = []
    for density, trunc in zip(densities_sample, truncated_reps):
        distances.append(quantile_distance(density, trunc) ** 2)
    mean_dist = np.mean(distances)
    return total_var - mean_dist