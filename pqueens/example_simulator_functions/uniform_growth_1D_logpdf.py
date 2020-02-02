import numpy as np

import pqueens.example_simulator_functions.uniform_growth_1D_lsq


def main(job_id, params):
    """ Interface to log of Gaussian likelihood of GnR model

    UNITS:
    - length [m]
    - mass [kg]
    - stress [Pa]

    Args:
        job_id (int):   ID of job
        params (dict):  Dictionary with parameters

    Returns:
        float:          Value of GnR model at parameters
                        specified in input dict
    """

    if 'k_sigma' not in params:
        params['k_sigma'] = pqueens.example_simulator_functions.uniform_growth_1D_lsq.K_SIGMA
    if 'sigma_h_c' not in params:
        params['sigma_h_c'] = pqueens.example_simulator_functions.uniform_growth_1D_lsq.SIGMA_H_C
    if 'dR0' not in params:
        params['dR0'] = pqueens.example_simulator_functions.uniform_growth_1D_lsq.DR0
    if 't0' not in params:
        params['t0'] = pqueens.example_simulator_functions.uniform_growth_1D_lsq.T0
    std_likelihood = pqueens.example_simulator_functions.uniform_growth_1D_lsq.STD_NOISE

    residuals = pqueens.example_simulator_functions.uniform_growth_1D_lsq.main(job_id, params)

    K1 = 2 * np.square(std_likelihood)
    log_likelihood = -1 / K1 * np.square(residuals).sum() - 0.5 * np.log(np.pi * K1)

    return log_likelihood
