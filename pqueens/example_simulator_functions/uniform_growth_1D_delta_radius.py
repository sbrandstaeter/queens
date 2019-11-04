import numpy as np

import pqueens.example_simulator_functions.uniform_growth_1D
import pqueens.example_simulator_functions.uniform_growth_1D_lsq

TAU = pqueens.example_simulator_functions.uniform_growth_1D_lsq.tau
K_SIGMA = pqueens.example_simulator_functions.uniform_growth_1D_lsq.k_sigma
SIGMA_H_C = pqueens.example_simulator_functions.uniform_growth_1D_lsq.sigma_h_c
DRO = pqueens.example_simulator_functions.uniform_growth_1D_lsq.dR0
T0 = pqueens.example_simulator_functions.uniform_growth_1D_lsq.t0

# point of time to evaluate the delta in radius
t = pqueens.example_simulator_functions.uniform_growth_1D_lsq.t_end


def main(job_id, params):
    """
    Interface to stability margin of GnR model.

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

    if 'tau' not in params:
        params['tau'] = TAU
    if 'k_sigma' not in params:
        params['k_sigma'] = K_SIGMA
    if 'sigma_h_c' not in params:
        params['sigma_h_c'] = SIGMA_H_C
    if 'dR0' not in params:
        params['dR0'] = DRO
    if 't0' not in params:
        params['t0'] = T0

    dR = pqueens.example_simulator_functions.uniform_growth_1D.delta_radius(
        t, params['tau'], params['k_sigma'], params['sigma_h_c'], params['dR0'], params['t0']
    )

    return dR
