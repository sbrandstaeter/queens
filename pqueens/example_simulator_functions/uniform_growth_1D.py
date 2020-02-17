"""
Uniform circumferential Growth and Remodelling of a homogeneous vessel

see Chapter 6.2 of [1].
In particular equations (78) + (80) with (79)

References:
    [1]: Cyron, C. J. and Humphrey, J. D. (2014)
    ‘Vascular homeostasis and the concept of mechanobiological stability’,
    International Journal of Engineering Science, 85, pp. 203–223.
    doi: 10.1016/j.ijengsci.2014.08.003.

"""

import numpy as np

# UNITS:
# - length [m]
# - mass [kg]
# - stress [Pa]


# NaN for uninitialized parameters
NaN = np.nan

# List of fixed model parameters
# initial thickness of aorta wall
H = 1.104761e-3

# mass fractions
phi_e = 0.293  # elastin
phi_m = 0.1897  # smooth muscle
phi_c = 0.5172  # collagen

# density
rho = 1050

# material density per unit area
M_e = phi_e * rho * H  # elastin
M_m = phi_m * rho * H  # smooth muscle
M_c = phi_c * rho * H  # collagen

# active stress contribution of smooth muscle
sigma_act_m = 30e3  # p.84 Kimmich2016

# correlation factor homeostatic stresses elastin and collagen
eta_m = 0.178  # p.84 Kimmich2016

# homeostatic smooth muscle stress (value from Christian's Matlab Code)
sigma_h_m = 6.704629134124602e03

# circumferential stress of elastin in initial configuration (value from Christian's Matlab Code)
sigma_cir_e = 1.088014923234574e05

# prestretch compared to natural configuration
G_e = 1.34  # elastin
G_m = 1.1  # smooth muscle
G_c = 1.062  # collagen

# elastic modulus (values taken from Christian's Matlab Code)
C_e = 0.325386455353085e06  # elastin
C_m = 0.168358396929622e06  # smooth muscle
C_c = 5.258871332154771e06  # collagen

# derived variables
# vector of mass fractions
Phi = np.array([phi_e, phi_m, phi_c])
# vector of material densities per unit surface area
M = np.array([M_e, M_m, M_c])
# vector of prestretches
G_ = np.array([G_e, G_m, G_c])
# vector of elastic modulus
C = np.array([C_e, C_m, C_c])
# initial radius
R = 1.25e-2

# sanity checks
# sum of mass fractions of all components should be equal to one
# np.sum(Phi)

# derived constants
K1 = M_m * C_m + M_c * C_c
K2 = -2 * (M_e * sigma_cir_e + M_m * sigma_h_m) + M_e * C_e
K3 = 2 * M_c
K4 = K1 + K2


def homeostatic_smooth_muscle_stress(sigma_h_c):
    """
    Return homeostatic stress of elastin.

    We use a linear correlation between homeostatic stress of collagen
    elastin.
    """
    sigma_h_m = sigma_act_m + eta_m * sigma_h_c
    return sigma_h_m


def stab_margin(tau, k_sigma, sigma_h_c, C=C):
    """
    Return stability margin.

    see eq. (79) in [1]
    """
    # circumferential stress in initial configuration
    sigma_cir = np.array([sigma_cir_e, sigma_h_m, sigma_h_c])

    # derived variables
    M_dot_sigma_cir = M.dot(sigma_cir)
    M_C = M * C

    # stability margin as in eq. (79) in [1]
    m_gnr = (tau * k_sigma * np.sum(M_C[1:]) - 2 * M_dot_sigma_cir + M_C[0]) / (
        tau * (np.sum(M_C) - 2 * M_dot_sigma_cir)
    )
    return m_gnr


def den_stab_margin(sigma_h_c):
    """
    Return the denominator of the stability margin.

    see denominator in eq. (79) in [1]
    """
    denominator = K4 - K3 * sigma_h_c
    return denominator


def numerator_stab_margin(tau, k_sigma, sigma_h_c):
    """
    Return the numerator of the stability margin.

    see numerator in eq. (79) in [1]
    """
    numerator = K1 * tau * k_sigma - K3 * sigma_h_c + K2
    return numerator


def numerator_stab_margin_is_zero(tau, sigma_h_c):
    """ Return k_sigma if the numerator of stability margin is zero. """
    k_sigma = (K3 * sigma_h_c - K2) / (K1 * tau)
    return k_sigma


def delta_radius(t, tau, k_sigma, sigma_h_c, dR0, t0):
    """
    Return delta of radius at time t.

    see eq. (3) with (78) + (79) in [1]
    """

    m_gnr = stab_margin(tau, k_sigma, sigma_h_c)
    dr = (1 + (tau * m_gnr - 1) * np.exp(-m_gnr * (t + t0))) * dR0 / (tau * m_gnr)
    return np.squeeze(dr)


def radius(tau, k_sigma, t, sigma_h_c, dR0, t0):
    """ Return current radius at time t. """
    r = R + delta_radius(t, tau, k_sigma, sigma_h_c, dR0, t0)
    return np.squeeze(r)


def main(job_id, params):
    """
    Interface to GnR model.

    Args:
        job_id (int):   ID of job
        params (dict):  Dictionary with parameters

    Returns:
        float:          Value of GnR model at parameters
                        specified in input dict
    """

    return delta_radius(
        params['t'],
        params['tau'],
        params['k_sigma'],
        params['sigma_h_c'],
        params['dR0'],
        params['t0'],
    )
