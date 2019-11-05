import numpy as np

import pqueens.example_simulator_functions.uniform_growth_1D
import pqueens.example_simulator_functions.uniform_growth_1D_lsq

TAU = pqueens.example_simulator_functions.uniform_growth_1D_lsq.tau
K_SIGMA = pqueens.example_simulator_functions.uniform_growth_1D_lsq.k_sigma
SIGMA_H_C = pqueens.example_simulator_functions.uniform_growth_1D_lsq.sigma_h_c

C_e = pqueens.example_simulator_functions.uniform_growth_1D.C_e
C_m = pqueens.example_simulator_functions.uniform_growth_1D.C_m
C_c = pqueens.example_simulator_functions.uniform_growth_1D.C_c

PHI_e = pqueens.example_simulator_functions.uniform_growth_1D.phi_e
PHI_m = pqueens.example_simulator_functions.uniform_growth_1D.phi_m
PHI_c = pqueens.example_simulator_functions.uniform_growth_1D.phi_c

H = pqueens.example_simulator_functions.uniform_growth_1D.H
RHO = pqueens.example_simulator_functions.uniform_growth_1D.rho

SIGMA_CIR_E = pqueens.example_simulator_functions.uniform_growth_1D.sigma_cir_e
SIGMA_H_M = pqueens.example_simulator_functions.uniform_growth_1D.sigma_h_m


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

    if 'C_e' not in params:
        params['C_e'] = C_e
    if 'C_m' not in params:
        params['C_m'] = C_m
    if 'C_c' not in params:
        params['C_c'] = C_c

    h = params.get("H", H)

    if 'phi_e' not in params:
        params['phi_e'] = PHI_e
    if 'phi_m' not in params:
        params['phi_m'] = PHI_m
    if 'phi_c' not in params:
        params['phi_c'] = PHI_c

    if 'sigma_cir_e' not in params:
        params['sigma_cir_e'] = SIGMA_CIR_E
    if 'sigma_h_m' not in params:
        params['sigma_h_m'] = SIGMA_H_M

    M = RHO * h * np.array([params['phi_e'], params['phi_m'], params['phi_c']])

    C = np.array([params['C_e'], params['C_m'], params['C_c']])

    stab_margin = pqueens.example_simulator_functions.uniform_growth_1D.stab_margin(
        params['tau'],
        params['k_sigma'],
        params['sigma_h_c'],
        C=C,
        M=M,
        sigma_cir_e=params['sigma_cir_e'],
        sigma_h_m=params['sigma_h_m'],
    )

    return stab_margin


if __name__ == "__main__":
    empty_dict = dict()
    print(main(0, empty_dict))
