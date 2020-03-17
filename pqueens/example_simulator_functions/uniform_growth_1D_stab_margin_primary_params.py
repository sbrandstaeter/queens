import numpy as np

# pylint: disable=line-too-long
import pqueens.example_simulator_functions.uniform_circumferential_growth_and_remodelling as uni_cir_gnr

# pylint: enable=line-too-long


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
    return uni_cir_gnr.UniformCircumferentialGrowthAndRemodellingParams(
        primary=True, **params
    ).m_gnr


if __name__ == "__main__":
    empty_dict = dict()
    print(main(0, empty_dict))
