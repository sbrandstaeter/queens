import numpy as np

import pqueens.example_simulator_functions.uniform_growth_1D as uniform_growth_1D

K1_CO = uniform_growth_1D.k1_co
K2_CO = uniform_growth_1D.k2_co
LAM_PRE_CO = uniform_growth_1D.lam_pre_co

K1_SM = uniform_growth_1D.k1_sm
K2_SM = uniform_growth_1D.k2_sm
LAM_PRE_SM = uniform_growth_1D.lam_pre_sm

MU_EL = uniform_growth_1D.mu_el
LAM_PRE_EL_CIR = uniform_growth_1D.lam_pre_el_cir
LAM_PRE_EL_AX = uniform_growth_1D.lam_pre_el_ax

PHI_e = uniform_growth_1D.phi_e
PHI_m = uniform_growth_1D.phi_m
PHI_c = uniform_growth_1D.phi_c

RHO = uniform_growth_1D.rho
R = uniform_growth_1D.R
MEAN_PRESSURE = uniform_growth_1D.mean_pressure


def main(job_id, params):
    """
    Interface to thickness of artery in homeostatic initial configuration based on Laplace law.

    UNITS:
    - length [m]
    - mass [kg]
    - stress [Pa]

    Args:
        job_id (int):   ID of job
        params (dict):  Dictionary with parameters

    Returns:
        float:          thickness such that artery is in homeostatic
                        state at parameters specified in input dict
    """

    k1_co = params.get("k1_co", K1_CO)
    k2_co = params.get("k2_co", K2_CO)
    lam_pre_co = params.get("lam_pre_co", LAM_PRE_CO)

    sigma_h_c = uniform_growth_1D.fung_cauchy_stress(lam_pre_co, k1_co, k2_co, rho=RHO)

    k1_sm = params.get("k1_sm", K1_SM)
    k2_sm = params.get("k2_sm", K2_SM)
    lam_pre_sm = params.get("lam_pre_sm", LAM_PRE_SM)

    sigma_h_m = uniform_growth_1D.fung_cauchy_stress(lam_pre_sm, k1_sm, k2_sm, rho=RHO)

    mu_el = params.get("mu_el", MU_EL)
    lam_pre_el_cir = params.get("lam_pre_el_cir", LAM_PRE_EL_CIR)
    lam_pre_el_ax = params.get("lam_pre_el_ax", LAM_PRE_EL_AX)

    sigma_cir_e = uniform_growth_1D.neo_hooke_cauchy_stress_cir(
        lam_pre_el_cir, lam_pre_el_ax, mu_el, rho=RHO
    )

    sigma = np.array([sigma_cir_e, sigma_h_m, sigma_h_c])

    phi_e = params.get("phi_e", PHI_e)
    phi_m = params.get("phi_m", PHI_m)
    phi_c = params.get("phi_c", PHI_c)

    phi = np.array([phi_e, phi_m, phi_c])

    mean_pressure = params.get("mean_pressure", MEAN_PRESSURE)

    # Laplace Law
    thickness = mean_pressure * R / (phi.dot(sigma))

    return thickness


if __name__ == "__main__":
    empty_dict = dict()
    print(main(0, empty_dict))
