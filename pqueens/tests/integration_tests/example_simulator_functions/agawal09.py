"""Agawal09 function.

[1]: Agarwal, N., & Aluru, N. R. (2009). A domain adaptive stochastic
     collocation approach for analysis of MEMS under uncertainties.
     Journal of Computational Physics, 228(20), 7662?7688.
     http://doi.org/10.1016/j.jcp.2009.07.014
"""
# pylint: disable=invalid-name

import numpy as np


def agawal09a(x1, x2, a1=0.5, a2=0.5, **kwargs):
    r"""Compute the Agawal09a function.

    Two dimensional benchmark funcion for UQ approaches proposed in [1].

    The function is defined as follows:

    :math:`f({\bf x}) = 0, \textrm{if } x_1 > \alpha_1 \textrm{ or } x_2 > \alpha_2`

    :math:`f({\bf x}) = \sin(\pi x_1 )\sin(\pi x_2 ), \textrm{ otherwise}`

    Distribution of the input random variables is probably uniform on [0,1]

    Args:
        x1 (float): first input parameter
        x2 (float): second input parameter
        a1 (float): coefficient (optional), with default value 0.5
        a2 (float): coefficient (optional), with default value 0.5

    Returns:
        float: value of agawal09a function
    """
    if x1 > a1 or x2 > a2:
        y = 0
    else:
        y = np.sin(np.pi * x1) * np.sin(np.pi * x2)

    return y
