"""Gradner2014a function is a two dimensional benchmark function for constraint
Bayesian optimization."""
import numpy as np


def gardner2014a(x1, x2):
    """Gradner2014a function: Two dimensional benchmark function for constraint
    Bayesian optimization [1]

    :math:`f({\\bf x}) = \\cos(2x_1)\\cos(x_2)+ \\sin(x_1)`

    with the corresponding constraint function:

    :math:`c({\\bf x}) = \\cos(x_1)\\cos(x_2) - \\sin(x_1)\sin(x_2)`

    with

    :math:`c({\\bf x}) \\leq 0.5`


    Args:
        x1 (float): input parameter 1 in [0, 6]
        x2 (float): input parameter 2 in [0, 6]

    Returns:
        float, float: value of gardner2014a function, value of corresponding
        constraint function


    References:

        [1] Gardner, Jacob R., Matt J. Kusner, Zhixiang Eddie Xu, Kilian Q.
            Weinberger, and John P. Cunningham. "Bayesian Optimization with
            Inequality Constraints." In ICML, pp. 937-945. 2014
    """

    y = np.cos(2 * x1) * np.cos(x2) + np.sin(x1)
    c = np.cos(x1) * np.cos(x2) - np.sin(x1) * np.sin(x2)

    return y, c


def main(job_id, params):
    """Interface to ma function.

    Args:
        job_id (int):  ID of job
        params (dict): Dictionary with parameters
    Returns:
        float, float: Value of gardner2014a function and constraint
                      at parameters specified in input dict
    """
    return gardner2014a(params['x1'], params['x2'])
