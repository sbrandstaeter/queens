from pqueens.tests.integration_tests.example_simulator_functions.branin_medfi import branin_medfi


def branin_lofi(x1, x2):
    """ Low-fidelity fidelity Branin function

    Compute value of medium-fidelity version of Branin function as described
    in [1]. The corresponding high- and low-fidelity versions are implemented
    in branin_hifi and branin_lofi, respectively.

    Args:
        x1 (float): first input parameter
        x2 (float): second input parameter

    Returns:
        float: Value of low-fidelity Branin function


    References:
        [1] Perdikaris, P. et al., 2017. Nonlinear information fusion algorithms
            for data-efficient multi-fidelity modelling.
            Proceedings of the Royal Society of London A: Mathematical,
            Physical and Engineering Sciences, 473(2198), pp.20160751–16.
    """
    y = branin_medfi(1.2 * (x1 + 2.0), 1.2 * (x2 + 2.0)) - 3.0 * x2 + 1.0

    return y


def main(job_id, params):
    """ Interface to low-fidelity Branin function
    Args:
        job_id (int):  ID of job
        params (dict): Dictionary with parameters

    Returns:
        float: Value of Branin function at parameters specified in input dict

    """
    return branin_lofi(params['x1'], params['x2'])
