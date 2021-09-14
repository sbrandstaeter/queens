import numpy as np


def sinus_test_fun(x1):
    """ A standard sinus as a test function

    Args:
        x1 (float): Input of the sinus in RAD

    Returns:
        float: Value of the sinus function

    """

    result = np.sin(x1)

    return result


def main(job_id, params):
    """ Interface to sinus test function

    Args:
        job_id (int):  ID of job
        params (dict): Dictionary with parameters

    Returns:
        float: Value of sinus test function at parameters specified in input dict
    """
    return sinus_test_fun(params['x1'])
