import numpy as np
from pqueens.example_simulator_functions.park91b_hifi import park91b_hifi

def park91b_lofi(x1, x2, x3, x4):
    """Low-fidelity version of Park91b benchmark function

    Simple four dimensional benchmark function as proposed in [1] to mimic
    a computer model. The low-fidelity version is defined as:

    :math:`f_{lofi}({\\bf x})=1.2 f_{hifi}({\\bf x})-1`

    The corresponding high-fidelity function is impelemented in park91b_hifi.

    Args:
        x1 (float): = Input parameter 1 [0,1)
        x2 (float): = Input parameter 2 [0,1)
        x3 (float): = Input parameter 3 [0,1)
        x4 (float): = Input parameter 4 [0,1)

    Returns:
        float: value of function at parameters

    References:
        [1] Park, J.-S.(1991). Tuning complex computer codes to data and optimal
            designs, Ph.D Thesis

        [2] Xiong, S., Qian, P., & Wu, C. (2013). Sequential design and analysis
            of high-accuracy and low-accuracy computer codes. Technometrics.
            http://doi.org/10.1080/00401706.2012.723572
    """

    yh = park91b_hifi(x1, x2, x3, x4)
    y = 1.2*yh - 1
    return y

def main(job_id, params):
    """ Interface to Park91b test fuction

    Args:
        job_id (int):  ID of job
        params (dict): Dictionary with parameters

    Returns:
        float: Value of the function at parameter specified in input dict
    """
    return park91b_lofi(params['x1'],params['x2'],params['x3'],params['x4'])
