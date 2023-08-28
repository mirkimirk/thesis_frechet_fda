"""This module contains functions for generating density samples in a simulation."""

import numpy as np

from misc import (
    trunc_norm_pdf,
    cdf_from_density,
    quantile_from_cdf,
    riemann_sum_arrays
)

def gen_grids_and_parameters(n, gridnum, truncation_point):
    """Generate parameters for the density samples and define appropriate grids."""

    grid_densities = np.linspace(
        start=-truncation_point,
        stop=truncation_point,
        num=gridnum,
    )
    grid_quantiles = np.linspace(start=0.001, stop=0.999, num=gridnum)

    # Draw different sigmas
    log_sigmas = np.random.default_rng(seed=28071995).uniform(-1.5, 1.5, n)
    mus = np.zeros(n)
    sigmas = np.exp(log_sigmas)

    return (
        grid_densities,
        grid_quantiles,
        mus,
        sigmas
    )


def gen_discretized_distributions(grid_pdfs, grid_qfs, mus, sigmas, truncation_point):
    """Generate discretized pdfs, cdfs, qfs, and qdfs."""
    # Truncated pdfs
    pdfs_discretized = trunc_norm_pdf(
        grid_pdfs[:, np.newaxis],
        mus,
        sigmas,
        -truncation_point,
        truncation_point,
    ).transpose()

    # Truncated cdfs
    cdfs_discretized = cdf_from_density(
        grid_pdfs,
        pdfs_discretized,
        axis=1,
    )

    # Truncated qfs
    qfs_discretized = quantile_from_cdf(
        grid_pdfs[:, np.newaxis].transpose(),
        cdfs_discretized,
        grid_qfs,
    )

    # Truncated qdfs
    qdfs_discretized = np.reciprocal(
        trunc_norm_pdf(
            qfs_discretized.transpose(),
            mus,
            sigmas,
            -truncation_point,
            truncation_point,
        ),
    ).transpose()

    # Normalize quantile densities
    qdfs_discretized = (
        qdfs_discretized
        * (grid_pdfs[-1] - grid_pdfs[0])
        / riemann_sum_arrays(grid_qfs, qdfs_discretized, axis = 1)[:, np.newaxis]
    )

    return pdfs_discretized, cdfs_discretized, qfs_discretized, qdfs_discretized