import numpy as np

# pylint: disable=line-too-long
import pqueens.example_simulator_functions.uniform_circumferential_growth_and_remodelling_radial_displacement_lsq as uni_cir_gnr_lsq

# pylint: enable=line-too-long


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

    std_likelihood = uni_cir_gnr_lsq.STD_NOISE

    res_sqrd = uni_cir_gnr_lsq.main(job_id, params)

    K1 = 2 * np.square(std_likelihood)
    log_likelihood = -1 / K1 * res_sqrd - 0.5 * np.log(np.pi * K1)

    return log_likelihood
