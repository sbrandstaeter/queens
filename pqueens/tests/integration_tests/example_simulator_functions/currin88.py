"""Currin functions."""
# pylint: disable=invalid-name

import numpy as np


def currin88_lofi(x1, x2, **kwargs):
    r"""Low-fidelity version of the Currin88 benchmark function.

    Simple two-dimensional example which appears several
    times in the literature, see, e.g., [1]-[3].

    The low-fidelity version is defined as follows [3]:

    :math:`f_{lofi}({\bf x}) = \frac{1}{4}[f_{hifi}(x_1+0.05,x_2+0.05) +
    f_{hifi}(x_1+0.05,max(0,x_2-0.05))] + \frac{1}{4}[f_{hifi}(x_1-0.05,x_2+0.05)
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


def currin88_hifi(x1, x2, **kwargs):
    r"""High-fidelity version of the Currin88 benchmark function.

    Simple two-dimensional example which appears several
    times in the literature, see, e.g., [1]-[3].

    The high-fidelity version is defined as follows:

    :math:`f_{hifi}({\bf x}) = [1 - \exp(\frac{1}{2 x_2})] \frac{2300x_1^3 +
    1900 x_1^2 + 2092x_1 +60}{100x_1^3 + 500x_1^2 +4 x_1+20}`

    [3] proposed the following low-fidelity version of the function:

    :math:`f_{lofi}({\bf x}) = \frac{1}{4}[f_{hifi}(x_1+0.05,x_2+0.05) +
    f_{hifi}(x_1+0.05,max(0,x_2-0.05))] + \frac{1}{4}[f_{hifi}(x_1-0.05,x_2+0.05)
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
            Technometrics, 55
        float: Value of currin88 function
    """
    fact1 = 1 - np.exp(-1 / (2 * x2))
    fact2 = 2300 * x1**3 + 1900 * x1**2 + 2092 * x1 + 60
    fact3 = 100 * x1**3 + 500 * x1**2 + 4 * x1 + 20

    y = fact1 * fact2 / fact3
    return y