import numpy as np


def borehole_lofi(rw, r, Tu, Hu, Tl, Hl, L, Kw):
    """Low-fidelity version of Borehole benchmark function.

    Very simple and quick to evaluate eight dimensional function that models
    water flow through a borehole. Frequently used function for testing a wide
    variety of methods in computer experiments.

    The low-fidelity version is defined as in [1] as:

    :math:`f_{lofi}({\\bf x}) = \\frac{5 T_u (H_u-H_l)}{ln(r/r_w)(1.5)
    + \\frac{2 L T_u}{ln(r/r_w)r_w^2 K_w} + \\frac{T_u}{T_l}}`

    For the purposes of uncertainty quantification, the distributions of the
    input random variables are often choosen as:

    | rw  ~ N(0.10,0.0161812)
    | r   ~ Lognormal(7.71,1.0056)
    | Tu  ~ Uniform[63070, 115600]
    | Hu  ~ Uniform[990, 1110]
    | Tl  ~ Uniform[63.1, 116]
    | Hl  ~ Uniform[700, 820]
    | L   ~ Uniform[1120, 1680]
    | Kw  ~ Uniform[9855, 12045]

    Args:
        rw (float): radius of borehole (m) [0.05, 0.15]
        r  (float): radius of influence (m) [100, 50000]
        Tu (float): transmissivity of upper aquifer (m2/yr) [63070, 115600]
        Hu (float): potentiometric head of upper aquifer (m)  [990, 1110]
        Tl (float): transmissivity of lower aquifer (m2/yr) [63.1, 116]
        Hl (float): potentiometric head of lower aquifer (m) [700, 820]
        L  (float): length of borehole (m)  [1120, 1680]
        Kw (float): hydraulic conductivity of borehole (m/yr)  [9855, 12045]

    Returns:
        float : The response is water flow rate, in m^3/yr.


    References:

        [1] Xiong, S., Qian, P. Z., & Wu, C. J. (2013). Sequential design and
            analysis of high-accuracy and low-accuracy computer codes.
            Technometrics, 55(1), 37-46.
    """

    frac1 = 5 * Tu * (Hu - Hl)

    frac2a = 2 * L * Tu / (np.log(r / rw) * rw ** 2 * Kw)
    frac2b = Tu / Tl
    frac2 = np.log(r / rw) * (1.5 + frac2a + frac2b)

    y = frac1 / frac2
    return y


def main(job_id, params):
    """Interface to low-fidelity borehole function.

    Args:
        job_id (int):  ID of job
        params (dict): Dictionary with parameters
    Returns:
        float: Value of borehole function at parameters specified in input dict
    """
    return borehole_lofi(
        params['rw'],
        params['r'],
        params['Tu'],
        params['Hu'],
        params['Tl'],
        params['Hl'],
        params['L'],
        params['Kw'],
    )
