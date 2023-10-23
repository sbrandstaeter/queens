"""Park91b functions."""
# pylint: disable=invalid-name

import numpy as np


def park91b_lofi(x1, x2, x3, x4, **kwargs):
    r"""Low-fidelity version of the Park91b benchmark function.

    Simple four-dimensional benchmark function as proposed in [1], to mimic
    a computer model. The low-fidelity version is defined as:

    :math:`f_{lofi}({\bf x})=1.2 f_{hifi}({\bf x})-1`

    The corresponding high-fidelity function is implemented in *park91b_hifi*.

    Args:
        x1 (float):  Input parameter 1 [0,1)
        x2 (float):  Input parameter 2 [0,1)
        x3 (float):  Input parameter 3 [0,1)
        x4 (float):  Input parameter 4 [0,1)

    Returns:
        float: Value of the function at the parameters

    References:
        [1] Park, J.-S.(1991). Tuning complex computer codes to data and optimal
            designs, Ph.D Thesis

        [2] Xiong, S., Qian, P., & Wu, C. (2013). Sequential design and analysis
            of high-accuracy and low-accuracy computer codes. Technometrics.
            http://doi.org/10.1080/00401706.2012.723572
    """
    yh = park91b_hifi(x1, x2, x3, x4)
    y = 1.2 * yh - 1
    return y


def park91b_hifi(x1, x2, x3, x4, **kwargs):
    r"""High-fidelity version of Park91b benchmark function.

    Simple four dimensional benchmark function as proposed in [1] to mimic
    a computer model. The high-fidelity version is defined as:

    :math:`f_{hifi}({\bf x})= \frac{2}{3} \exp(x_1 + x_2) - x_4 \sin(x_3) + x_3`

    For the purpose of multi-fidelity simulation, [2] defined a corresponding
    lower fidelity function, which is  implemented in *park91b_lofi*.

    Args:
        x1 (float):  Input parameter 1 [0,1)
        x2 (float):  Input parameter 2 [0,1)
        x3 (float):  Input parameter 3 [0,1)
        x4 (float):  Input parameter 4 [0,1)

    Returns:
        float: Value of function at parameters

    References:
        [1] Park, J.-S.(1991). Tuning complex computer codes to data and optimal
            designs, Ph.D Thesis

        [2] Xiong, S., Qian, P., & Wu, C. (2013). Sequential design and analysis
            of high-accuracy and low-accuracy computer codes. Technometrics.
            http://doi.org/10.1080/00401706.2012.723572
    """
    term1 = (2 / 3) * np.exp(x1 + x2)
    term2 = -x4 * np.sin(x3)
    term3 = x3

    y = term1 + term2 + term3
    return y
