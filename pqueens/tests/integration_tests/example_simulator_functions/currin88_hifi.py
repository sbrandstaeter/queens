import numpy as np


def currin88_hifi(x1, x2):
    """ High-fidelity version of the Currin88 benchmark function

    Simple two-dimensional example which appears several
    times in the literature, see, e.g., [1]-[3].

    The high-fidelity version is defined as follows:

    :math:`f_{hifi}({\\bf x}) = [1 - \\exp(\\frac{1}{2 x_2})] \\frac{2300x_1^3 +
    1900 x_1^2 + 2092x_1 +60}{100x_1^3 + 500x_1^2 +4 x_1+20}`

    [3] proposed the following low-fidelity version of the function:

    :math:`f_{lofi}({\\bf x}) = \\frac{1}{4}[f_{hifi}(x_1+0.05,x_2+0.05) +
    f_{hifi}(x_1+0.05,max(0,x_2-0.05))] + \\frac{1}{4}[f_{hifi}(x_1-0.05,x_2+0.05)
    + f_{hifi}(x_1-0.05,max(0,x_2-0.05))]`


    Args:
        x1 (float): input parameter 1 in [0,1]
        x2 (float): input parameter 2 in [0,1]


    Returns:
        float: value of currin88 function

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

    fact1 = 1 - np.exp(-1 / (2 * x2))
    fact2 = 2300 * x1 ** 3 + 1900 * x1 ** 2 + 2092 * x1 + 60
    fact3 = 100 * x1 ** 3 + 500 * x1 ** 2 + 4 * x1 + 20

    y = fact1 * fact2 / fact3
    return y


def main(job_id, params):
    """ Interface to currin88 function

    Args:
        job_id (int):  ID of job
        params (dict): Dictionary with parameters
    Returns:
        float: Value of currin88 function at parameters specified in input dict
    """
    return currin88_hifi(params['x1'], params['x2'])
