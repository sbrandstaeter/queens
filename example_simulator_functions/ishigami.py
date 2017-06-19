import numpy as np

def ishigami(x1, x2, x3, p1=None, p2=None):
    # TODO look at the documentation
    """ Three dimensional benchmark function

    Three dimensional benchmark function from `[2]`_ used for UQ because it
    exhibits it exhibit strong nonlinearity and nonmonotonicity.
    It also has a peculiar dependence on x3, as described `[5]`_).
    The values of a and b used in `[1]`_ and `[3]`_ are: p_1 = a = 7
    and p_2 = b = 0.1. In `[5]`_ the values a = 7 and b = 0.05 are used.


     :math:`f({\bf x}) = \sin(x_1) + p1 \sin^2(x_2 +p2x_3^4\sin(x_1))`

    Typically distributions of the input random variables are is:
    :math:`x_i ~` Uniform[:math:`-\pi, \pi`], for all i = 1, 2, 3

    References:

    .. _[1] Crestaux, T., Martinez, J.-M., Le Maitre, O., & Lafitte, O. (2007).
            Polynomial chaos expansion for uncertainties quantification and
            sensitivity analysis [PowerPoint slides]. Retrieved from
            SAMO 2007 website: http://samo2007.chem.elte.hu/lectures/Crestaux.pdf.

    .. _[2] Ishigami, T., & Homma, T. (1990, December). An importance
            quantification technique in uncertainty analysis for computer models.
            In Uncertainty Modeling and Analysis, 1990. Proceedings.,

    .. _[3] Marrel, A., Iooss, B., Laurent, B., & Roustant, O. (2009).
            Calculations of sobol indices for the gaussian process metamodel.
            Reliability Engineering & System Safety, 94(3), 742-751.

    .. _[4] Saltelli, A., Chan, K., & Scott, E. M. (Eds.). (2000).
            Sensitivity analysis (Vol. 134). New York: Wiley.

    .. _[5] Sobol, I. M., & Levitan, Y. L. (1999). On the use of variance
            reducing multipliers in Monte Carlo computations of a global
            sensitivity index. Computer Physics Communications, 117(1), 52-61.


    Args:

    x1 (float):
    x2 (float):
    x3 (float):
    p1 (float): coefficient (optional), with default value 7
    p2 (float): coefficient (optional), with default value 0.1

    Returns:
        float : value of ishigami function at [x1, x2, x3]
    """

    if p1 is None:
        p1 = 7
    if p2 is None:
        p2 = 0.1


    term1 = np.sin(x1)
    term2 = p1 * (np.sin(x2))**2
    term3 = p2 * x3**4 * np.sin(x1)

    return term1 + term2 + term3

def main(job_id, params):
    return ishigami(params['x1'], params['x2'], params['x3'])