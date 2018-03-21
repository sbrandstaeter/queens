import numpy as np
from pqueens.example_simulator_functions.perdikaris_1dsin_lofi import perdikaris_1dsin_lofi


def perdikaris_1dsin_hifi(x):
    """ High-fidelity version of simple 1-d test function

    High-fideltiy version of simple 1-dimensional benchmark function as
    proposed in [1] and defined as:

    :math:`f_{hifi}(x)= (x-\\sqrt{2})(f_{lofi}(x))^2`

    The low-fidelity version of the function was also prodposed in [1]
    and is in implemented in perdikaris_1dsin_lofi

    Args:

     x (float): Input parameter [0,1]
    Returns:
        float: Value of function at x

    References:
        [1] Perdikaris, P. et al., 2017. Nonlinear information fusion algorithms
            for data-efficient multi-fidelity modelling.
            Proceedings of the Royal Society of London A: Mathematical,
            Physical and Engineering Sciences,  473(2198), pp.20160751?16.
    """
    y = (x - np.sqrt(2))*perdikaris_1dsin_lofi(x)**2
    return y

def main(job_id, params):
    """ Interface to Perdikaris test fuction

    Args:
        job_id (int):  ID of job
        params (dict): Dictionary with parameters

    Returns:
        float: Value of the function at parameter specified in input dict
    """
    return perdikaris_1dsin_hifi(params['x'])
