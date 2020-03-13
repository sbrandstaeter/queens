import numpy as np

# pylint: disable=line-too-long
import pqueens.example_simulator_functions.uniform_circumferential_growth_and_remodelling as uniform_growth_1D

# pylint: enable=line-too-long

days_to_seconds = 1  # 86400
years_to_days = 365.25
years_to_seconds = years_to_days * days_to_seconds

# real parameter values
TAU = 100 * days_to_seconds
# k_sigma = 0.001107733872774 / days_to_seconds
K_SIGMA = 4.933441e-04 / days_to_seconds
SIGMA_H_C = 200000
DR0 = 0.002418963148596
T0 = 2 * years_to_seconds


T_END = 3000 * days_to_seconds

NUM_MEAS = 8

# t_meas = np.linspace(-t0, t_end, num_meas)
T_MEAS = np.linspace(0.0, T_END, NUM_MEAS)
# t_meas = [0, 730, 1460, 2190]
M_GNR = uniform_growth_1D.stab_margin(TAU, K_SIGMA, SIGMA_H_C)
DR_MEAS = uniform_growth_1D.delta_radius(T_MEAS, TAU, M_GNR, DR0, T0)

RNG_STATE = np.random.get_state()
SEED = 24
np.random.seed(SEED)
STD_NOISE = 1.0e-3  # 1.0e-2
NOISE = np.random.normal(0, STD_NOISE, NUM_MEAS)

np.random.set_state(RNG_STATE)

DR_MEAS = DR_MEAS + NOISE

import pqueens.example_simulator_functions.uniform_growth_1D_stab_margin as uniform_growth_1D_stab_margin


def main(job_id, params):
    """ Interface to least squares of GnR model

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
    # make sure that tau for m_gnr and for delta_radius is equal
    params["tau"] = tau
    dR0 = params.get("dR0", DR0)
    t0 = params.get("t0", T0)

    m_gnr = uniform_growth_1D_stab_margin.main(job_id, params)
    dr_sim = uniform_growth_1D.delta_radius(t=T_MEAS, tau=tau, m_gnr=m_gnr, dR0=dR0, t0=t0)

    residuals = DR_MEAS - dr_sim

    return np.square(residuals).sum()


if __name__ == "__main__":
    # test with default parameters
    empty_dict = dict()
    print(main(0, empty_dict))
