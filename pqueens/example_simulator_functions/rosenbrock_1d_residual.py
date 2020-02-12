from pqueens.example_simulator_functions.rosenbrock_residual import rosenbrock_residual


def main(job_id, params):
    """ Interface to Residuals of Rosenbrock banana function

    Args:
        job_id (int):   ID of job
        params (dict):  Dictionary with parameters

    Returns:
        ndarray: Vector of residuals of the Rosenbrock function at the
                 positions specified in the params dict
    """
    return rosenbrock_residual(params['x1'], x2=1.0)
