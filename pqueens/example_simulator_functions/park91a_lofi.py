import numpy as np
from pqueens.example_simulator_functions.park91a_hifi import park91a_hifi

def park91a_lofi(x1, x2, x3, x4):
    """ Low-fidelity Park91a function

    Simple four dimensional benchmark function as proposed in [1] to mimic
    a computer model. For the purpose of multi-fidelity simulation, [3]
    defined a corresponding lower fidelity function, which is  defined as

    :math: `f({\\bf x}) = \\frac{x_1}{2}[\\sqrt{1+(x_2+x_3^2)\\frac{x_4}{x_1^2}}-1]+(x_1+3x_4)\\exp[1-\sin(x_3)]`

    The high-fidelity version is defiend as is implemented in park91a_hifi

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

    [2] Cox, D. D., Park, J.-S., & Singer, C. E. (2001). A statistical method
        for tuning a computer code to a data base. Computational Statistics &
        Data Analysis, 37(1), 77?92. http://doi.org/10.1016/S0167-9473(00)00057-8

    [3] Xiong, S., Qian, P., & Wu, C. (2013). Sequential design and analysis of
        high-accuracy and low-accuracy computer codes. Technometrics.
        http://doi.org/10.1080/00401706.2012.723572

    """
    yh = park91a_hifi(x1, x2, x3, x4)
    term1 = (1+np.sin(x1)/10) * yh
    term2 = -2*x1 + x2**2 + x3**2
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
    return park91a_lofi(params['x1'],params['x2'],params['x3'],params['x4'])
