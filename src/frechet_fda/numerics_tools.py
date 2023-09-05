"""This module contains an implementation of the Riemann sum to approximate
integrals, as well as the difference quotient to approximate a derivative."""

import numpy as np


def riemann_sum_cumulative(
    x_vals: np.ndarray, y_vals: np.ndarray, method: str = "left", axis: int = -1,
) -> tuple:
    """Computes cumulative integral.

    Can take arrays for x_vals and y_vals, assumes last axis to hold the grid points
    which are to integrate.

    """
    # Get distances between points
    point_distance = np.diff(x_vals)
    # Initialize integral array
    integral = np.zeros_like(y_vals)
    if method == "left":
        integral[..., 1:] = np.cumsum(y_vals[..., :-1] * point_distance, axis = axis)
    elif method == "right":
        integral[..., 1:] = np.cumsum(y_vals[..., 1:] * point_distance, axis = axis)
    elif method == "midpoint":
        y_midpoints = (y_vals[:-1] + y_vals[1:]) / 2
        integral[..., 1:] = np.cumsum(y_midpoints * point_distance, axis = axis)
    else:
        msg = "Method must specify either left, right, or midpoint Riemann sum!"
        raise ValueError(msg)
    return x_vals, integral


def difference_quotient(x_vals: np.ndarray, y_vals: np.ndarray):
    """Computes derivative of discretized function."""
    d_x = x_vals
    d_y = np.zeros_like(y_vals)
    d_y[:-1] = np.diff(y_vals) / np.diff(d_x)
    d_y[-1] = d_y[-2]
    return d_x, d_y
