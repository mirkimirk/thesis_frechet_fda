"""This module contains functions for generating density samples in a simulation."""

import numpy as np
from misc import (
    cdf_from_density,
    qdf_from_density,
    quantile_from_density,
    trunc_norm_pdf,
)


def gen_grids_and_parameters(n, gridnum, truncation_point, delta):
    """Generate parameters for the density samples and define appropriate grids."""
    grid_densities = np.linspace(
        start=-truncation_point,
        stop=truncation_point,
        num=gridnum,
    )
    grid_quantiles = np.linspace(start=delta, stop=1 - delta, num=gridnum)

    # Draw different sigmas
    log_sigmas = np.random.default_rng(seed=28071995).uniform(-1.5, 1.5, n)
    mus = np.zeros(n)
    sigmas = np.exp(log_sigmas)

    return (grid_densities, grid_quantiles, mus, sigmas)


def gen_discretized_distributions(grid_pdfs, grid_qfs, mus, sigmas, truncation_point):
    """Generate discretized pdfs, cdfs, qfs, and qdfs."""
    # Truncated pdfs
    pdfs_discretized = trunc_norm_pdf(
        grid_pdfs[:, np.newaxis],
        mus,
        sigmas,
        -truncation_point,
        truncation_point,
    )

    # Truncated cdfs
    cdfs_discretized = cdf_from_density(
        grid_pdfs,
        pdfs_discretized,
        axis=-1,
    )

    # Truncated qfs
    qfs_discretized = quantile_from_density(
        pdfs_discretized,
        grid_pdfs,
        grid_qfs,
    )

    # Truncated qdfs
    qdfs_discretized = qdf_from_density(
        pdfs_discretized, dsup=grid_pdfs, qdsup=grid_qfs,
    )

    return pdfs_discretized, cdfs_discretized, qfs_discretized, qdfs_discretized
