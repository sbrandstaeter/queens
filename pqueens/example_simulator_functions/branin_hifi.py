import numpy as np

def branin_hifi(x1, x2):
    """ High-fidelity Branin function

    Compute value of high fidelity version of Branin function as described
    in [1]. The corresponding medium- and low-fidelity versions are implemented
    in branin_medfi and branin_lofi, respectively.

    Args:
        x1 (float): first input parameter [−5, 10]
        x2 (float): second input parameter [0, 15]

    Returns:
        float: value of high-fidelity Branin function

    References:
        [1] Perdikaris, P. et al., 2017. Nonlinear information fusion algorithms
            for data-efficient multi-fidelity modelling.
            Proceedings of the Royal Society of London A: Mathematical,
            Physical and Engineering Sciences, 473(2198), pp.20160751–16.
    """

    result = (-1.275*x1**2 / np.pi**2 + 5.0*x1/np.pi + x2 - 6.0)**2 + \
             (10.0 - 5.0/(4.0*np.pi))*np.cos(x1) + 10.0

    return result

def main(job_id, params):
    """ Interface to high-fidelity Branin function

    Args:
        job_id (int):  ID of job
        params (dict): Dictionary with parameters

    Returns:
        float: Value of Branin function at parameters specified in input dict
    """
    return branin_hifi(params['x1'], params['x2'])


