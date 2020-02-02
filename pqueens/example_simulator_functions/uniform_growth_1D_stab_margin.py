import numpy as np

import pqueens.example_simulator_functions.uniform_growth_1D as uniform_growth_1D
import pqueens.example_simulator_functions.uniform_growth_1D_lsq as uniform_growth_1D_lsq



TAU = uniform_growth_1D_lsq.TAU
K_SIGMA = uniform_growth_1D_lsq.K_SIGMA
SIGMA_H_C = uniform_growth_1D_lsq.SIGMA_H_C

C_E = uniform_growth_1D.C_e
C_M = uniform_growth_1D.C_m
C_C = uniform_growth_1D.C_c

PHI_e = uniform_growth_1D.phi_e
PHI_m = uniform_growth_1D.phi_m
PHI_c = uniform_growth_1D.phi_c

H = uniform_growth_1D.H
RHO = uniform_growth_1D.rho

SIGMA_CIR_E = uniform_growth_1D.sigma_cir_e
SIGMA_H_M = uniform_growth_1D.sigma_h_m


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
    tau = params.get("tau", TAU)
    k_sigma = params.get("k_sigma", K_SIGMA)
    sigma_h_c = params.get("sigma_h_c", SIGMA_H_C)

    C_e = params.get("C_e", C_E)
    C_m = params.get("C_m", C_M)
    C_c = params.get("C_c", C_C)

    h = params.get("H", H)

    phi_e = params.get("phi_e", PHI_e)
    phi_m = params.get("phi_m", PHI_m)
    phi_c = params.get("phi_c", PHI_c)

    sigma_cir_e = params.get("sigma_cir_e", SIGMA_CIR_E)
    sigma_h_m = params.get("sigma_h_m", SIGMA_H_M)

    M = RHO * h * np.array([phi_e, phi_m, phi_c])

    C = np.array([C_e, C_m, C_c])

    stab_margin = uniform_growth_1D.stab_margin(
        tau,
        k_sigma,
        sigma_h_c,
        C=C,
        M=M,
        sigma_cir_e=sigma_cir_e,
        sigma_h_m=sigma_h_m,
    )

    return stab_margin


if __name__ == "__main__":
    empty_dict = dict()
    print(main(0, empty_dict))
