import numpy as np

# pylint: disable=line-too-long
import pqueens.example_simulator_functions.uniform_circumferential_growth_and_remodelling as uni_cir_gnr

# pylint: enable=line-too-long


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

    # Laplace Law
    thickness = uni_cir_gnr.UniformCircumferentialGrowthAndRemodellingParams(
        primary=True, **params
    ).h

    return thickness
