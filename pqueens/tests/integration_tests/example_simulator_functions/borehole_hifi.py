import numpy as np


def borehole_hifi(rw, r, Tu, Hu, Tl, Hl, L, Kw):
    """High-fidelity version of Borehole benchmark function.

    Very simple and quick to evaluate eight dimensional function that models
    water flow through a borehole. Frequently used function for testing a wide
    variety of methods in computer experiments, see, e.g., [1]-[10].

    The high-fidelity version is defined as:

    :math:`f_{hifi}({\\bf x}) = \\frac{2 \\pi T_u(H_u-H_l)}{ln(r/r_w)
    (1 + \\frac{2LT_u}{ln(r/r_w)r_w^2K_w})+ \\frac{T_u}{T_l}}`

    For the purpose of multi-fidelity simulation, Xiong et al. (2013) [8] use
    the following function for the lower fidelity code:

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

        [1] An, J., & Owen, A. (2001). Quasi-regression. Journal of Complexity,
            17(4), 588-607.

        [2] Gramacy, R. B., & Lian, H. (2012). Gaussian process single-index models
            as emulators for computer experiments. Technometrics, 54(1), 30-41.

        [3] Harper, W. V., & Gupta, S. K. (1983). Sensitivity/uncertainty analysis
            of a borehole scenario comparing Latin Hypercube Sampling and
            deterministic sensitivity approaches (No. BMI/ONWI-516).
            Battelle Memorial Inst., Columbus, OH (USA). Office of Nuclear
            Waste Isolation.

        [4] Joseph, V. R., Hung, Y., & Sudjianto, A. (2008). Blind kriging:
            A new method for developing metamodels. Journal of mechanical design,
            130, 031102.

        [5] Moon, H. (2010). Design and Analysis of Computer Experiments for
            Screening Input Variables (Doctoral dissertation, Ohio State University).

        [6] Moon, H., Dean, A. M., & Santner, T. J. (2012). Two-stage
            sensitivity-based group screening in computer experiments.
            Technometrics, 54(4), 376-387.

        [7] Morris, M. D., Mitchell, T. J., & Ylvisaker, D. (1993).
            Bayesian design and analysis of computer experiments: use of derivatives
            in surface prediction. Technometrics, 35(3), 243-255.

        [8] Xiong, S., Qian, P. Z., & Wu, C. J. (2013). Sequential design and
            analysis of high-accuracy and low-accuracy computer codes.
            Technometrics, 55(1), 37-46.

        [9] Worley, B. A. (1987). Deterministic uncertainty analysis
            (No. CONF-871101-30). Oak Ridge National Lab., TN (USA).

        [10] Zhou, Q., Qian, P. Z., & Zhou, S. (2011). A simple approach to
             emulation for computer models with qualitative and quantitative
             factors. Technometrics, 53(3).

    for further information, see also http://www.sfu.ca/~ssurjano/borehole.html
    """

    frac1 = 2 * np.pi * Tu * (Hu - Hl)

    frac2a = 2 * L * Tu / (np.log(r / rw) * rw ** 2 * Kw)
    frac2b = Tu / Tl
    frac2 = np.log(r / rw) * (1 + frac2a + frac2b)

    y = frac1 / frac2
    return y


def main(job_id, params):
    """Interface to high-fidelity borehole function.

    Args:
        job_id (int):  ID of job
        params (dict): Dictionary with parameters
    Returns:
        float: Value of borehole function at parameters specified in input dict
    """
    return borehole_hifi(
        params['rw'],
        params['r'],
        params['Tu'],
        params['Hu'],
        params['Tl'],
        params['Hl'],
        params['L'],
        params['Kw'],
    )
