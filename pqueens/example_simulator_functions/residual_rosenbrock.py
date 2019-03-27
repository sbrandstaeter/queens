import numpy as np

def residual_rosenbrock(x1, x2):
    """
    Residual of Rosenbrock banana function

    Args:
        x1 (float):  Input one
        x2 (float):  Input two

    Returns:
        float: Value of Rosenbrock function
    """

    a = 10* (x2 - x1*x1)
    b = 1. - x1

    return np.array([a, b])

def main(job_id, params):
    """ Interface to Rosenbrock Banana function

    Args:
        job_id (int):   ID of job
        params (dict):  Dictionary with parameters

    Returns:
        float:          Value of Rosenbrock function at parameters
                        specified in input dict
    """
    return residual_rosenbrock(params['x1'], params['x2'])
