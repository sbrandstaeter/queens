import numpy as np
from pqueens.tests.integration_tests.example_simulator_functions.park91a_hifi import park91a_hifi


def park91a_lofi_coords(x1, x2, x3, x4):
    # catch values outside of definition
    if x1 <= 0:
        x1 = 0.01
    elif x1 >= 1:
        x1 = 0.99

    if x2 <= 0:
        x2 = 0.01
    elif x2 >= 1:
        x2 = 0.99

    if x3 <= 0:
        x3 = 0.01
    elif x3 >= 1:
        x3 = 0.99

    if x4 <= 0:
        x4 = 0.01
    elif x4 >= 1:
        x4 = 0.99

    yh = park91a_hifi(x1, x2, x3, x4)
    term1 = (1 + np.sin(x1) / 10) * yh
    term2 = -2 * x1 + x2 ** 2 + x3 ** 2
    y = term1 + term2 + 0.5

    return y


def main(job_id, params):
    """ Interface to Park91a test fuction

    Args:
        job_id (int):  ID of job
        params (dict): Dictionary with parameters

    Returns:
        float: Value of the function at parameter specified in input dict
    """

    # use x3 and x4 as coordinates and create coordinate grid
    xx3 = np.linspace(0, 1, 4)
    xx4 = np.linspace(0, 1, 4)
    x3_vec, x4_vec = np.meshgrid(xx3, xx4)
    x3_vec = x3_vec.flatten()
    x4_vec = x4_vec.flatten()

    # evaluate testing functions for coordinates and fixed input
    y_vec = []
    for x3, x4 in zip(x3_vec, x4_vec):
        y_vec.append(park91a_lofi_coords(params['x1'], params['x2'], x3, x4))
    y_vec = np.array(y_vec)

    return y_vec
