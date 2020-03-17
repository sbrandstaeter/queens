import numpy as np

import pqueens.example_simulator_functions.uniform_growth_1D_lsq as uniform_growth_1d_lsq


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

    std_likelihood = uniform_growth_1d_lsq.STD_NOISE

    res_sqrd = uniform_growth_1d_lsq.main(job_id, params)

    K1 = 2 * np.square(std_likelihood)
    log_likelihood = -1 / K1 * res_sqrd - 0.5 * np.log(np.pi * K1)

    return log_likelihood
