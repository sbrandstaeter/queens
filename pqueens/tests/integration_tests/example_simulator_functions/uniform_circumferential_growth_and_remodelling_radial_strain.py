# pylint: disable=line-too-long
import pqueens.tests.integration_tests.example_simulator_functions.uniform_circumferential_growth_and_remodelling as uni_cir_gnr
import pqueens.tests.integration_tests.example_simulator_functions.uniform_circumferential_growth_and_remodelling_radial_displacement_lsq as uniform_growth_1D_lsq

# pylint: enable=line-too-long

# default point of time to evaluate the delta in radius
T = uniform_growth_1D_lsq.T_END


def main(job_id, params):
    """Interface to radial engineering strain of GnR model parameterized with
    primary parameters.

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

    # current time to evaluate growth at
    t = params.pop("t", T)

    gnr_model = uni_cir_gnr.UniformCircumferentialGrowthAndRemodelling(primary=True, **params)
    return gnr_model.de_r(t)
