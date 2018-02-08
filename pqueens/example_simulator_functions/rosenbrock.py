def rosenbrock(x1, x2):
    """ Rosenbrock banana function """

    a = 1. - x1
    b = x2 - x1*x1
    return a*a + b*b*100.

def main(job_id, params):
    """ Interface to Rosenbrock Banana function

    Args:
        job_id (int):   ID of job
        params (dict):  Dictionary with parameters

    Returns:
        float:          Value of Rosenbrock function at parameters
                        specified in input dict
    """
    return rosenbrock(params['x1'], params['x2'])
