"""This module contains the Function class."""


import matplotlib.pyplot as plt
import numpy as np

from frechet_fda.numerics_tools import difference_quotient, riemann_sum_cumulative


class Function:
    """Contains methods intended for functions that characterize distributions,
    converting one into the other, computing log transformations etc.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y
        self.grid_size = len(x)

    def standardize_shape(self, grid_size: int = 1000):
        """Set the number of discretization points."""
        x = np.copy(self.x)
        y = np.copy(self.y)
        new_x = np.linspace(x.min(), x.max(), grid_size)
        new_y = np.interp(new_x, x, y)
        return Function(new_x, new_y)

    def warp_range(self, left: float, right: float, grid_size: int = 1000):
        """Use interpolation to define function on different range."""
        x = np.copy(self.x)
        y = np.copy(self.y)
        new_x = np.linspace(left, right, grid_size)
        new_y = np.interp(new_x, x, y)
        return Function(new_x, new_y)

    def integrate(self, limits: tuple = None, method: str = "left"):
        """Integrate function using Riemann sums.

        Either `left`, `right`, or `midpoint` rule is used.

        """
        if limits is None:
            x = np.copy(self.x)
            y = np.copy(self.y)
        else:
            x = np.copy(self.x)[(self.x >= limits[0]) & (self.x <= limits[1])]
            y = np.copy(self.y)[(self.x >= limits[0]) & (self.x <= limits[1])]

        int_x, int_y = riemann_sum_cumulative(x_vals=x, y_vals=y, method=method)
        return Function(int_x, int_y)

    def integrate_sequential(self):
        """Legacy method for integration."""
        x = np.copy(self.x)
        y = np.copy(self.y)

        int_x = x
        point_distance = np.diff(x)
        int_y = np.zeros_like(y)
        for i in range(1, x.shape[0]):
            int_y[i] = int_y[i - 1] + y[i - 1] * point_distance[i - 1]
        return Function(int_x, int_y)

    def differentiate(self):
        """Give derivative of function."""
        x = np.copy(self.x)
        y = np.copy(self.y)

        d_x, d_y = difference_quotient(x_vals=x, y_vals=y)
        return Function(d_x, d_y)

    def invert(self):
        """Compute inverse."""
        inv_x = np.copy(self.y)
        inv_y = np.copy(self.x)
        return Function(inv_x, inv_y)

    def log(self):
        """Log transform the function."""
        log_x = np.copy(self.x)
        log_y = np.log(np.copy(self.y))
        return Function(log_x, log_y)

    def exp(self):
        """Exponential transform the function."""
        exp_x = np.copy(self.x)
        exp_y = np.exp(np.copy(self.y))
        return Function(exp_x, exp_y)

    def compose(self, other):
        """Compute function composition with another."""
        comp_x = np.copy(other.x)
        comp_y = np.interp(other.y, self.x, self.y)
        return Function(comp_x, comp_y)

    def plot(self, restricted=0, xlims: tuple = None):
        """Plot the function values against their support."""
        if restricted > 0:
            rest_x = self.x[restricted:-restricted]
            rest_y = self.y[restricted:-restricted]
        else:
            rest_x = self.x
            rest_y = self.y
        ax = plt.subplots()[1]
        plt.plot(rest_x, rest_y)
        plt.grid()
        # Set x-axis limits
        if xlims is not None:
            ax.set_xlim(xlims[0], xlims[1])
        # Boldly highlight the y=0 line
        ax.axhline(0, color="black", linewidth=0.7)
        plt.show()

    def compare(
        self,
        other,
        xlims: tuple = None,
        label_self: str = "self",
        label_other: str = "other",
    ):
        """Compare the shapes of the functions in a plot."""
        ax = plt.subplots()[1]
        ax.plot(self.x, self.y, color="C0", label=label_self)
        ax.plot(other.x, other.y, color="C1", label=label_other)
        if xlims is not None:
            ax.set_xlim(xlims[0], xlims[1])
        plt.legend()
        plt.grid()
        plt.show()

    def vcenter(self):
        """Shift the range of values after integration of the qdf, to reflect correct
        support.
        """
        x = np.copy(self.x)
        y = np.copy(self.y)
        return Function(x, y - np.mean(y))

    def regularize(self):
        """Make infs to huge or tiny numbers."""
        x = np.copy(self.x)
        y = np.copy(self.y)
        y = np.nan_to_num(y, copy=False)
        return Function(x, y)

    def l2norm(self):
        """Compute L2 norm of (approximate) function."""
        x = np.copy(self.x)
        y = np.copy(self.y)
        return np.sqrt(riemann_sum_cumulative(x_vals=x, y_vals=y**2)[1][..., -1])

    def __add__(self, val: float | int):
        x = np.copy(self.x)
        y = np.copy(self.y)
        if isinstance(val, float | int):
            return Function(x, y + val)
        elif isinstance(val, Function):
            left = min(x[0], val.x[0])
            right = max(x[-1], val.x[-1])
            # Define grid_size, take the finer number of the functions that are to add
            gridnum_self = len(x)
            gridnum_other = len(val.x)
            grid_size = max(gridnum_self, gridnum_other)
            comb_x = np.linspace(left, right, grid_size)
            comb_y = np.interp(comb_x, x, y) + np.interp(comb_x, val.x, val.y)
            return Function(comb_x, comb_y)

    def __sub__(self, val: float | int):
        x = np.copy(self.x)
        y = np.copy(self.y)
        if isinstance(val, float | int):
            return Function(x, y - val)
        elif isinstance(val, Function):
            left = min(x[0], val.x[0])
            right = max(x[-1], val.x[-1])
            # Define grid_size, take the finer number of the functions that are to add
            gridnum_self = len(x)
            gridnum_other = len(val.x)
            grid_size = max(gridnum_self, gridnum_other)
            comb_x = np.linspace(left, right, grid_size)
            comb_y = np.interp(comb_x, x, y) - np.interp(comb_x, val.x, val.y)
            return Function(comb_x, comb_y)

    def __mul__(self, val: float | int):
        x = np.copy(self.x)
        y = np.copy(self.y) * val
        return Function(x, y)

    def __rmul__(self, val: float | int):
        x = np.copy(self.x)
        y = val * np.copy(self.y)
        return Function(x, y)

    def __truediv__(self, val: float | int):
        x = np.copy(self.x)
        y = np.copy(self.y) / val
        return Function(x, y)

    def __rtruediv__(self, val: float | int):
        x = np.copy(self.x)
        y = val / np.copy(self.y)
        return Function(x, y)

    def __pow__(self, exponent: float | int):
        if not isinstance(exponent, float | int):
            raise TypeError("Exponent must be a float or an integer.")
        x = np.copy(self.x)
        y = np.copy(self.y) ** exponent
        return Function(x, y)

    def __neg__(self):
        x = np.copy(self.x)
        y = -np.copy(self.y)
        return Function(x, y)
