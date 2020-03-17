import numpy as np

# pylint: disable=line-too-long
import pqueens.example_simulator_functions.uniform_circumferential_growth_and_remodelling as uni_cir_gnr

# pylint: enable=line-too-long

years_to_days = 365.25

# real parameter values
T0 = 2 * years_to_days
PARAMS = {"t0": T0}

T_END = 3000

NUM_MEAS = 8

# t_meas = np.linspace(-t0, t_end, num_meas)
T_MEAS = np.linspace(0.0, T_END, NUM_MEAS)
GNR_MODEL = uni_cir_gnr.UniformCircumferentialGrowthAndRemodelling(primary=True, **PARAMS)
DR_MEAS = GNR_MODEL.dr(t=T_MEAS)

# create random noise
RNG_STATE = np.random.get_state()
SEED = 24
np.random.seed(SEED)
STD_NOISE = 1.0e-3  # 1.0e-2
NOISE = np.random.normal(0, STD_NOISE, NUM_MEAS)

np.random.set_state(RNG_STATE)

# add random noise on pseudo measurements
DR_MEAS = DR_MEAS + NOISE


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

    # make sure the default t0 value is set correctly
    t0 = params.get("t0", T0)
    params["t0"] = t0
    gnr_model = uni_cir_gnr.UniformCircumferentialGrowthAndRemodelling(primary=True, **params)

    dr_sim = gnr_model.dr(t=T_MEAS)

    residuals = DR_MEAS - dr_sim

    return np.square(residuals).sum()
