import numpy as np

def perdikaris_1dsin_lofi(x):
    """ Low-fidelity version of simple 1-d test function

    Low-fideltiy version of simple 1-dimensional benchmark function as
    proposed in [1] and defined as:

    :math:`f_{lofi}({\\bf x}) = \\sin(8.0\\pi x)`

    The high-fidelity version of the function was also prodposed in [1]
    and is in implemented in perdikaris_1dsin_hifi

    Args:

     x (float): = Input parameter
    Returns:
        float: value of function at x

    References:
        [1] Perdikaris, P. et al., 2017. Nonlinear information fusion algorithms
            for data-efficient multi-fidelity modelling.
            Proceedings of the Royal Society of London A: Mathematical,
            Physical and Engineering Sciences,  473(2198), pp.20160751?16.
    """
    y =  np.sin(8.0*np.pi*x)
    return y

def main(job_id, params):
    """ Interface to Perdikaris test fuction

    Args:
        job_id (int):  ID of job
        params (dict): Dictionary with parameters

    Returns:
        float: Value of the function at parameter specified in input dict
    """
    return perdikaris_1dsin_lofi(params['x'])
