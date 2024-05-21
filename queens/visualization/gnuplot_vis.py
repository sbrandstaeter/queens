"""Gnuplot visualization."""

import logging

import numpy as np

_logger = logging.getLogger(__name__)
try:
    from gnuplotlib import plot

except FileNotFoundError:
    _logger.warning("Cannot import gnuplotlib, no terminal plots available.")

    # Gnuplot is not available on certain system
    def plot(*_args, **_kwargs):
        """Dummy function if no gnuplot is available."""


def gnuplot_gp_convergence(iter_lst, fun_value_lst):
    """Convergence plots for Gaussian Process optimization and convergence.

    Args:
        iter_lst (lst): List with iteration numbers up to now
        fun_value_lst (lst): List with values of a function
    """
    plot(
        np.array(iter_lst).reshape(1, -1),
        np.array(fun_value_lst).reshape(1, -1),
        unset="grid",
        terminal="dumb 60,30",
        _with="lines",
    )
