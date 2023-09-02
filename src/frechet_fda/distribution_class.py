import warnings

import matplotlib.pyplot as plt
import numpy as np

from frechet_fda.numerics_helpers import riemann_sum_cumulative, difference_quotient


class Distribution:
    """Contains methods intended for functions that characterize distributions,
    converting one into the other, computing log transformations etc.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y


    def integrate(self, method: str = "left"):
        """Integrate function using Riemann sums.

        Either `left`, `right`, or `midpoint` rule is used.

        """
        x = np.copy(self.x)
        y = np.copy(self.y)

        int_x, int_y = riemann_sum_cumulative(x_vals=x, y_vals=y, method=method)
        return Distribution(int_x, int_y)


    def integrate_sequential(self):
        x = np.copy(self.x)
        y = np.copy(self.y)

        int_x = x
        point_distance = np.diff(x)
        int_y = np.zeros_like(y)
        for i in range(1, x.shape[0]):
            int_y[i] = int_y[i - 1] + y[i - 1] * point_distance[i - 1]
        return Distribution(int_x, int_y)


    def differentiate(self):
        x = np.copy(self.x)
        y = np.copy(self.y)

        d_x, d_y = difference_quotient(x_vals = x, y_vals = y)
        return Distribution(d_x, d_y)


    def invert(self):
        inv_x = np.copy(self.y)
        inv_y = np.copy(self.x)
        return Distribution(inv_x, inv_y)


    def log(self):
        log_x = np.copy(self.x)
        log_y = np.log(np.copy(self.y))
        return Distribution(log_x, log_y)


    def exp(self):
        exp_x = np.copy(self.x)
        exp_y = np.exp(np.copy(self.y))
        return Distribution(exp_x, exp_y)


    def compose(self, other):
        comp_x = np.copy(other.x)
        comp_y = np.interp(other.y, self.x, self.y)
        return Distribution(comp_x, comp_y)


    def plot(self, restricted=0):
        if restricted > 0:
            rest_x = self.x[restricted:-restricted]
            rest_y = self.y[restricted:-restricted]
        else:
            rest_x = self.x
            rest_y = self.y
        fig, ax = plt.subplots()
        plt.plot(rest_x,rest_y)
        plt.grid()
        # Boldly highlight the y=0 line
        ax.axhline(0, color='black', linewidth=0.7)
        plt.show()


    def compare(self, other, label_self: str = "self", label_other: str = "other"):
        fig, ax = plt.subplots()
        ax.plot(self.x, self.y, color="C0", label=label_self)
        ax.plot(other.x, other.y, color="C1", label=label_other)
        plt.legend()
        plt.grid()
        plt.show()

    
    def vcenter(self):
        x = np.copy(self.x)
        y = np.copy(self.y)
        return Distribution(x,y-np.mean(y))


    def __sub__(self, val: float | int):
        x = np.copy(self.x)
        y = np.copy(self.y)
        if isinstance(val, float | int):
            return Distribution(x, y - val)
        elif isinstance(val, Distribution):
            return Distribution(x, y - val.y)


    def __add__(self, val: float | int):
        x = np.copy(self.x)
        y = np.copy(self.y)
        if isinstance(val, float | int):
            return Distribution(x, y + val)
        elif isinstance(val, Distribution):
            return Distribution(x, y + val.y)
