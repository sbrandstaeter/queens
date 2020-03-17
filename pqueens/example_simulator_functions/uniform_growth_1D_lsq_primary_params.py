import numpy as np

# pylint: disable=line-too-long
import pqueens.example_simulator_functions.uniform_circumferential_growth_and_remodelling as uni_cir_gnr

# pylint: enable=line-too-long
import pqueens.example_simulator_functions.uniform_growth_1D_lsq as uniform_growth_1D_lsq


# import real parameter values and measurements
T0 = uniform_growth_1D_lsq.T0
PARAMS = {"t0": T0}

T_MEAS = uniform_growth_1D_lsq.T_MEAS
DE_MEAS = uni_cir_gnr.UniformCircumferentialGrowthAndRemodelling(
    primary=True, **PARAMS
).delta_radius(t=T_MEAS)

DE_MEAS = DE_MEAS + uniform_growth_1D_lsq.NOISE


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
    de_sim = uni_cir_gnr.UniformCircumferentialGrowthAndRemodelling(
        primary=True, **params
    ).delta_radius(t=T_MEAS)
    residuals = DE_MEAS - de_sim

    return np.square(residuals).sum()


if __name__ == "__main__":
    # test with default parameters
    empty_dict = dict()
    print(main(0, empty_dict))
