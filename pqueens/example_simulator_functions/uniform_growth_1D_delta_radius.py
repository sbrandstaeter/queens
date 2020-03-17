# pylint: disable=line-too-long
import pqueens.example_simulator_functions.uniform_circumferential_growth_and_remodelling as uni_cir_gnr

# pylint: enable=line-too-long
import pqueens.example_simulator_functions.uniform_growth_1D_lsq as uniform_growth_1D_lsq

# default point of time to evaluate the delta in radius
T = uniform_growth_1D_lsq.T_END


def main(job_id, params):
    """
    Interface to engineering strain of radius of GnR model parameterized with primary parameters.

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
    t = params.get("t", T)

    de_r = uni_cir_gnr.UniformCircumferentialGrowthAndRemodelling(
        primary=False, **params
    ).delta_radius(t)

    return de_r


if __name__ == "__main__":
    empty_dict = dict()
    print(main(0, empty_dict))
