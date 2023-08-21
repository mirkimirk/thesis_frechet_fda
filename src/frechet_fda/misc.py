"""Contains various functions needed in different modules."""
import numpy as np


def riemann_sum(a, b, f, method="midpoint", step_size=None):
    """Vectorized approximate integral."""
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


def riemann_sum_arrays(left_bound, right_bound, multi_dim_array, axis):
    """"Computes riemann sum for given array, along the axis that contains the grid of
    values.
    """
    m = multi_dim_array.shape[axis]  # Number of points along the axis of grid values
    step_size = (right_bound - left_bound) / m

    # Compute the Riemann sum along the axis of grid values using vectorized computation
    return np.sum(multi_dim_array, axis=axis) * step_size


def l2_norm(left_bound_support, right_bound_support, array, axis):
    """Compute L2 norm of (approximate) function."""
    return np.sqrt(
        riemann_sum_arrays(
            left_bound=left_bound_support,
            right_bound=right_bound_support,
            multi_dim_array=array**2,
            axis=axis,
        ),
    )


# def wasserstein_norm(left_bound_support, right_bound_support, array, axis):
#     """Compute Wasserstein norm."""
# To Do!!
