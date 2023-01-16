"""TODO_doc."""

import gnuplotlib as gnp
import numpy as np


def gnuplot_gp_convergence(iter_lst, fun_value_lst):
    """TODO_doc: this is not in the documentation.

    Make some convergence plots for Gaussian Process optimization and
    convergence.

    Args:
        iter_lst (lst): List with iteration numbers up to now
        fun_value_lst (lst): List with values of a function
    """
    gnp.plot(
        np.array(iter_lst).reshape(1, -1),
        np.array(fun_value_lst).reshape(1, -1),
        unset='grid',
        terminal='dumb 60,30',
        _with='lines',
    )
