"""Contains various functions needed in different modules."""
import numpy as np


def riemann_sum(a, b, f, method="midpoint", step_size=None):
    """Approximate integral."""
    if step_size is None:
        step_size = (b - a) / 1000
        m = int((b - a) / step_size)
    if method == "left":
        grid = np.linspace(a, b - step_size, m)
    elif method == "right":
        grid = np.linspace(a + step_size, b, m)
    elif method == "midpoint":
        grid = np.linspace(a, b, m + 1)
        grid = (grid[1:] + grid[:-1]) / 2
    else:
        msg = "Must specify either left, right, or midpoint Riemann sum!"
        raise ValueError(msg)
    return np.sum(f(grid) * step_size)
