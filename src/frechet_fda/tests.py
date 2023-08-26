"""Contains some tests."""

import numpy as np

from misc import (
    l2_norm
)


def test_norm_eigenfunctions():
    # Check the first few L^2 norms to verify
    norms = l2_norm(
        support_grid=grid_densities_univ,
        array=eigenfunctions_scaled,
        axis=1,
        cumsum=False,
    )
    assert np.allclose(norms, 1, atol=1e-8)
test_norm_eigenfunctions()