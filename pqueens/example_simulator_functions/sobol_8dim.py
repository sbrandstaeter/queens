def sobol(x1, x2, x3, x4, x5, x6, x7, x8, a=None):
    """ Compute eight-dimensional Sobol function

    Args:
        x_i (float): Input in range [0,1]

    """


    if a is None:
        a = [0, 1, 4.5, 9, 99, 99, 99, 99]

    output = 1
    values = [x1, x2, x3, x4, x5, x6, x7, x8]

    for j in range(8):
        x = values[j]
        output *= (abs(4 * x - 2) + a[j]) / (1 + a[j])

    return output

def main(job_id, params):
    """ Interface to eight-dimensional Sobol G function

    Args:
        job_id (int):   ID of job
        params (dict):  Dictionary with parameters

    Returns:
        float:          Value of eight dimensional sobol G function at parameters
                        specified in input dict
    """
    return sobol(params['x1'], params['x2'], params['x3'], params['x4'],
                 params['x5'], params['x6'], params['x7'], params['x8'])
