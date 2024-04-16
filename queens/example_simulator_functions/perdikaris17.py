"""Perdikaris test functions."""
import numpy as np


def perdikaris17_lofi(x):
    r"""Low-fidelity version of simple 1D test function.

    Low-fidelity version of a simple 1-dimensional benchmark function as
    proposed in [1] and defined as:

    :math:`f_{lofi}({\bf x}) = \sin(8.0\pi x)`

    The high-fidelity version of the function was also proposed in [1]
    and is in implemented in *perdikaris_1dsin_hifi*.

    Args:
        x (float): Input parameter

    Returns:
        float: Value of function at *x*

    References:
        [1] Perdikaris, P. et al., 2017. Nonlinear information fusion algorithms
            for data-efficient multi-fidelity modelling.
            Proceedings of the Royal Society of London A: Mathematical,
            Physical and Engineering Sciences,  473(2198), pp.20160751?16.
    """
    y = np.sin(8.0 * np.pi * x)
    return y


def perdikaris17_hifi(x):
    r"""High-fidelity version of simple 1D test function.

    High-fidelity version of simple 1-dimensional benchmark function as
    proposed in [1] and defined as:

    :math:`f_{hifi}(x)= (x-\sqrt{2})(f_{lofi}(x))^2`

    The low-fidelity version of the function was also proposed in [1]
    and is in implemented in *perdikaris_1dsin_lofi*.

    Args:
        x (float): Input parameter [0,1]

    Returns:
        float: Value of function at *x*

    References:
        [1] Perdikaris, P. et al., 2017. Nonlinear information fusion algorithms
            for data-efficient multi-fidelity modelling.
            Proceedings of the Royal Society of London A: Mathematical,
            Physical and Engineering Sciences,  473(2198), pp.20160751?16.
    """
    y = (x - np.sqrt(2)) * perdikaris17_lofi(x) ** 2
    return y
