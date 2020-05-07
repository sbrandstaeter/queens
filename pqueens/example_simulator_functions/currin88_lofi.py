import numpy as np
from pqueens.example_simulator_functions.currin88_hifi import currin88_hifi


def currin88_lofi(x1, x2):
    """ low-fidelity version of the Currin88 benchmark function

    Simple two-dimensional example which appears several
    times in the literature, see, e.g., [1]-[3].

    The low-fidelity version is defined as follows [3]:

    :math:`f_{lofi}({\\bf x}) = \\frac{1}{4}[f_{hifi}(x_1+0.05,x_2+0.05) +
    f_{hifi}(x_1+0.05,max(0,x_2-0.05))] + \\frac{1}{4}[f_{hifi}(x_1-0.05,x_2+0.05)
    + f_{hifi}(x_1-0.05,max(0,x_2-0.05))]`


    Args:
        x1 (float): input parameter 1 in [0,1]
        x2 (float): input parameter 2 in [0,1]


    Returns:
        float: Value of currin88 function

    References:

        [1] Currin, C., Mitchell, T., Morris, M., & Ylvisaker, D. (1988).
            A Bayesian approach to the design and analysis of computer
            experiments. Technical Report 6498. Oak Ridge National Laboratory.

        [2] Currin, C., Mitchell, T., Morris, M., & Ylvisaker, D. (1991).
            Bayesian prediction of deterministic functions, with applications
            to the design and analysis of computer experiments. Journal of the
            American Statistical Association, 86(416), 953-963.

        [3] Xiong, S., Qian, P. Z., & Wu, C. J. (2013). Sequential design and
            analysis of high-accuracy and low-accuracy computer codes.
            Technometrics, 55(1), 37-46.
    """

    # maxarg = np.maximum(np.zeros((1,len(x2)),dtype = float), x2-1/20)
    maxarg = np.maximum(np.zeros((1, 1), dtype=float), x2 - 1 / 20)

    yh1 = currin88_hifi(x1 + 1 / 20, x2 + 1 / 20)
    yh2 = currin88_hifi(x1 + 1 / 20, maxarg)
    yh3 = currin88_hifi(x1 - 1 / 20, x2 + 1 / 20)
    yh4 = currin88_hifi(x1 - 1 / 20, maxarg)

    y = (yh1 + yh2 + yh3 + yh4) / 4
    return y


def main(job_id, params):
    """ Interface to currin88 function

    Args:
        job_id (int):  ID of job
        params (dict): Dictionary with parameters
    Returns:
        float: Value of currin88 function at parameters specified in input dict
    """
    return currin88_lofi(params['x1'], params['x2'])
