"""High-fidelity Park91a function with x3 and x4 as fixed coordinates."""
import numpy as np


def park91a_hifi_coords(x1, x2, x3, x4):
    r"""High-fidelity Park91a function.

    High-fidelity Park91a function with x3 and x4 as fixed coordinates.
    Coordinates are prescribed in the main function of this module.

    Simple four dimensional benchmark function as proposed in [1] to mimic
    a computer model. For the purpose of multi-fidelity simulation, [3]
    defined a corresponding lower fidelity function, which is  implemented
    in park91a_lofi.

    The high-fidelity version is defined as:

    :math:`f({\\bf x}) =
    \\frac{x_1}{2}[\\sqrt{1+(x_2+x_3^2)\\frac{x_4}{x_1^2}}-1]+(x_1+3x_4)\\exp[1-\\sin(x_3)]`

    Args:
        x1 (float): Input parameter 1 [0,1)
        x2 (float): Input parameter 2 [0,1)
        x3 (float): Input parameter 3 [0,1)
        x4 (float): Input parameter 4 [0,1)

    Returns:
        float: Value of function at parameters

    References:
        [1] Park, J.-S.(1991). Tuning complex computer codes to data and optimal
            designs, Ph.D. Thesis

        [2] Cox, D. D., Park, J.-S., & Singer, C. E. (2001). A statistical method
            for tuning a computer code to a database. Computational Statistics &
            Data Analysis, 37(1), 77?92. http://doi.org/10.1016/S0167-9473(00)00057-8

        [3] Xiong, S., Qian, P., & Wu, C. (2013). Sequential design and analysis of
            high-accuracy and low-accuracy computer codes. Technometrics.
            http://doi.org/10.1080/00401706.2012.723572
    """
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

    term1a = x1 / 2
    term1b = np.sqrt(1 + (x2 + x3 ** 2) * x4 / (x1 ** 2)) - 1
    term1 = term1a * term1b

    term2a = x1 + 3 * x4
    term2b = np.exp(1 + np.sin(x3))
    term2 = term2a * term2b

    y = term1 + term2

    # ----
    term1a = x1 / 2
    d_term1a_dx1 = 1 / 2
    term1b = np.sqrt(1 + (x2 + x3 ** 2) * x4 / (x1 ** 2)) - 1
    d_term1b_dx1 = (
        1
        / (2 * np.sqrt(1 + (x2 + x3 ** 2) * x4 / (x1 ** 2)))
        * (-2 * (x2 + x3 ** 2) * x4 * x1 ** (-3))
    )
    term1 = term1a * term1b
    d_term1_dx1 = d_term1a_dx1 * term1b + term1a * d_term1b_dx1

    term2a = x1 + 3 * x4
    d_term2a_dx1 = 1
    term2b = np.exp(1 + np.sin(x3))
    d_term2b_dx1 = 0
    term2 = term2a * term2b
    d_term2_dx1 = d_term2a_dx1 * term2b + term2a * d_term2b_dx1

    dy_dx1 = d_term1_dx1 + d_term2_dx1

    # ----
    term1a = x1 / 2
    d_term1a_dx2 = 0
    term1b = np.sqrt(1 + (x2 + x3 ** 2) * x4 / (x1 ** 2)) - 1
    d_term1b_dx2 = 1 / (2 * np.sqrt(1 + (x2 + x3 ** 2) * x4 / (x1 ** 2))) * x4 / (x1 ** 2)
    term1 = term1a * term1b
    d_term1_dx2 = d_term1a_dx2 * term1b + term1a * d_term1b_dx2

    term2a = x1 + 3 * x4
    term2b = np.exp(1 + np.sin(x3))
    term2 = term2a * term2b
    d_term2_dx2 = 0

    dy_dx2 = d_term1_dx2 + d_term2_dx2

    return y, (dy_dx1, dy_dx2)


def main(_job_id, params):
    """Interface to Park91a test function.

    Args:
        _job_id (int):  ID of job
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
    y_grad = []
    for x3, x4 in zip(x3_vec, x4_vec):
        y_vec.append(park91a_hifi_coords(params['x1'], params['x2'], x3, x4)[0])
        y_grad.append(park91a_hifi_coords(params['x1'], params['x2'], x3, x4)[1][:])
    y_vec = np.array(y_vec)
    y_grad = np.array(y_grad)
    return y_vec, y_grad
