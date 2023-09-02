"""This module contains tools to use with the distribution class."""

import numpy as np
from frechet_fda.distribution_class import Distribution


def make_distribution_objects(
        x_grid : np.ndarray, y_array: np.ndarray
    ) -> list[Distribution]:
    """Takes an x range and possibly multiple densitites defined on it in the y_array.
    Assumes that rows of """
    return [Distribution(x_grid, y_grid) for y_grid in y_array]
