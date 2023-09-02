import warnings

import matplotlib.pyplot as plt
import numpy as np


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

        int_x, int_y = _riemann_sum_cumulative(x_vals=x, y_vals=y, method=method)
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

        d_x = x
        d_y = np.zeros_like(y)
        d_y[:-1] = np.diff(y) / np.diff(x)
        d_y[-1] = d_y[-2]
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
        plt.plot(rest_x, rest_y)
        plt.grid()

    def compare(self, other, label_self: str, label_other: str):
        fig, ax = plt.subplots()
        ax.plot(self.x, self.y, color="C0", label=label_self)
        ax.plot(other.x, other.y, color="C1", label=label_other)
        plt.legend()
        plt.grid()
        plt.show()

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


def make_pdf(
    left: float,
    right: float,
    mu: np.ndarray = 0,
    sigma: np.ndarray = 1,
    grid_size: int = 1000,
) -> list[Distribution]:
    pdf_x = np.linspace(left, right, grid_size)
    pdfs_y = trunc_norm_pdf(
        pdf_x[:, np.newaxis],
        mu,
        sigma,
        left,
        right,
    )
    return [Distribution(pdf_x, pdf_y) for pdf_y in pdfs_y]


def gen_params_scenario_one(
    num_of_distr: int,
) -> tuple:
    """Generate parameters for the density samples and define appropriate grids."""
    # Draw different sigmas
    log_sigmas = np.random.default_rng(seed=28071995).uniform(-1.5, 1.5, num_of_distr)
    mus = np.zeros(num_of_distr)
    sigmas = np.exp(log_sigmas)

    return mus, sigmas


# Truncated normal pdf
def trunc_norm_pdf(
    x: np.ndarray, mu: np.ndarray, sigma: np.ndarray, a: float, b: float,
):
    """Define truncated normal density function.

    To test: columns of x must align with mu and sigma.

    """
    x = np.array(x)  # to vectorize the input
    mu = np.array(mu)
    sigma = np.array(sigma)
    x_std = (x - mu) / sigma
    a_std = (a - mu) / sigma
    b_std = (b - mu) / sigma
    numerator = _norm_pdf(x_std, 0, 1)
    denominator = _norm_cdf(b_std, 0, 1) - _norm_cdf(a_std, 0, 1)

    result = numerator / denominator / sigma

    # Create a boolean mask for values outside the interval [a, b]
    mask = (x_std < a_std) | (x_std > b_std)

    # Set the PDF to zero for values of x outside the interval [a, b]
    result[mask] = 0
    result = result.transpose()

    # Check whether each density integrates to 1
    eps = 1e-5
    integrals = _riemann_sum_cumulative(np.linspace(a, b, len(x)), result, axis=-1)[1]
    deviations_from_1 = abs(integrals - 1)
    if np.any(deviations_from_1 > eps):
        warnings.warn(
            f"Not all provided densities integrate to 1 with tolerance {eps}!"
            f"\n Max case of deviation is: {deviations_from_1.max()}"
            f"\n In position: {deviations_from_1.argmax()}"
            "\n Performing normalization...",
        )
        result /= integrals[..., -1]
    return result


# Normal pdf
def _norm_pdf(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    """Define normal density function.

    To test: columns of x must align with mu and sigma.

    """
    x = np.array(x)  # to vectorize the input
    mu = np.array(mu)
    sigma = np.array(sigma)
    return np.reciprocal(np.sqrt(2 * np.pi) * sigma) * np.exp(
        (-0.5) * ((x - mu) / sigma) ** 2,
    )


# Normal cdf
def _norm_cdf(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Compute the CDF of the normal distribution at a given point x.

    `x` can be a 2d array, with the number of rows corresponding to the number of
    densities n. Accordingly, `mu` and `sigma` can be vectors of length n.

    """
    a = -20  # Lower limit of integration (approximation of negative infinity)
    grid_to_integrate = np.linspace(a, x, 20000)
    # Integrate the normal density function from a to b
    return _riemann_sum_cumulative(
        grid_to_integrate, _norm_pdf(grid_to_integrate, mu, sigma),
    )[1][..., -1]


def _riemann_sum_cumulative(
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
        integral[..., 1:] = np.cumsum(y_vals[..., :-1] * point_distance, axis=axis)
    elif method == "right":
        integral[..., 1:] = np.cumsum(y_vals[..., 1:] * point_distance, axis=axis)
    elif method == "midpoint":
        y_midpoints = (y_vals[:-1] + y_vals[1:]) / 2
        integral[..., 1:] = np.cumsum(y_midpoints * point_distance, axis=axis)
    else:
        msg = "Method must specify either left, right, or midpoint Riemann sum!"
        raise ValueError(msg)
    return x_vals, integral
