"""Module to test external simulator functions."""
from pqueens.tests.integration_tests.example_simulator_functions.ishigami90 import ishigami90


def main(_job_id, params):
    """Main function.

    Args:
        _job_id (int):  ID of job
        params (dict): Dictionary with parameters

    Returns:
        float: Value of the function at parameter specified in input dict
    """
    return ishigami90(**params)
