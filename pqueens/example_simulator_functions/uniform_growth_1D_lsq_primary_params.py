import numpy as np

# pylint: disable=line-too-long
import pqueens.example_simulator_functions.uniform_circumferential_growth_and_remodelling as uniform_growth_1D
import pqueens.example_simulator_functions.uniform_growth_1D_lsq as uniform_growth_1D_lsq
import pqueens.example_simulator_functions.uniform_growth_1D_stab_margin_primary_params as uniform_growth_1D_stab_margin_primary_params

# pylint: enable=line-too-long

# import real parameter values and measurements
TAU = uniform_growth_1D_lsq.TAU
DR0 = uniform_growth_1D_lsq.DR0
T0 = uniform_growth_1D_lsq.T0

T_MEAS = uniform_growth_1D_lsq.T_MEAS
# TODO: calculate measurements based on primary parameters
DR_MEAS = uniform_growth_1D_lsq.DR_MEAS


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

    m_gnr = uniform_growth_1D_stab_margin_primary_params.main(job_id, params)
    dr_sim = uniform_growth_1D.delta_radius(t=T_MEAS, tau=tau, m_gnr=m_gnr, dR0=dR0, t0=t0)

    residuals = DR_MEAS - dr_sim

    return np.square(residuals).sum()


if __name__ == "__main__":
    # test with default parameters
    empty_dict = dict()
    print(main(0, empty_dict))
