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
# - time [d]

# NaN for uninitialized parameters
NaN = np.nan

# List of fixed model parameters
# initial radius
R0 = 1.25e-2
# initial thickness of aorta wall
H = 1.104761e-3

years_to_days = 365.25
# time of initial damage (in the past: T0 > 0)
T0 = 2 * years_to_days

# initial damage (engineering strain at t=t0)
DR0 = 0.002418963148596

# mean blood pressure in Pascal (see [1] Section 6.2 p.219)
mmhg_to_pa = 101325 / 760  # see wikipedia article Torr=mmHg
MEAN_PRESSURE = 93 * mmhg_to_pa

# mass fractions
PHI_EL = 0.293  # elastin
PHI_SM = 0.1897  # smooth muscle
PHI_CO = 0.5172  # collagen

# density
RHO = 1050

# turnover time collagen & smooth muscle
TAU = 70
# gain parameter collagen & smooth muscle
K_SIGMA = 4.933441e-04

# material parameters of collagen
K1_CO = 568  # Fung parameter 1 collagen
K2_CO = 11.2  # Fung parameter 2 collagen

# material parameters of smooth muscle
K1_SM = 7.6  # Fung parameter 1 smooth muscle
K2_SM = 11.4  # Fung parameter 2 smooth muscle

# material parameters of elastin matrix
MU_EL = 72  # (3D) Neo-Hooke shear parameter

# homeostatic collagen stress
SIGMA_H_CO = 200000

# homeostatic smooth muscle stress (value from Christian's Matlab Code)
SIGMA_H_SM = 6.704629134124602e03

# circumferential stress of elastin in initial configuration (value from Christian's Matlab Code)
SIGMA_CIR_EL = 1.088014923234574e05

# prestretch compared to natural configuration
LAM_PRE_EL_CIR = 1.34  # elastin circumferential
LAM_PRE_EL_AX = 1.25  # elastin axial
LAM_PRE_SM = 1.1  # smooth muscle
# lam_pre_co = 1.062  # collagen
LAM_PRE_CO = 1.060697162749238753320924e00  # collagen BACI value

# NOTE: G and C codepend via other material parameters (see Christian's Matlab Code)
# elastic modulus (values taken from Christian's Matlab Code)
C_EL = 0.325386455353085e06  # elastin
C_SM = 0.168358396929622e06  # smooth muscle
# C_c = 5.391358494528612e06  # collagen (corresponds to lam_pre_co = 1.062)
C_CO = 5.258871332154771e06  # collagen (corresponds to lam_pre_co=1.060697162...)

# derived variables
# material density per unit area
M_e = PHI_EL * RHO * H  # elastin
M_m = PHI_SM * RHO * H  # smooth muscle
M_c = PHI_CO * RHO * H  # collagen
# vector of mass fractions
Phi = np.array([PHI_EL, PHI_SM, PHI_CO])
# vector of material densities per unit surface area
M = np.array([M_e, M_m, M_c])
# vector of prestretches
G_ = np.array([LAM_PRE_EL_CIR, LAM_PRE_SM, LAM_PRE_CO])
# vector of elastic modulus
C = np.array([C_EL, C_SM, C_CO])


def fung_cauchy_stress(lambda_e, k1, k2, rho=1.0):
    lam_e_sqrd = lambda_e ** 2
    lam_e_sqrd_one = lam_e_sqrd - 1
    return 2 * rho * k1 * lam_e_sqrd * lam_e_sqrd_one * np.exp(k2 * lam_e_sqrd_one ** 2)


def fung_elastic_modulus(lambda_e, k1, k2, rho=1.0):
    lam_e_sqrd = lambda_e ** 2
    lam_e_sqrd_one = lam_e_sqrd - 1
    lam_e_sqrd_one_sqrd = lam_e_sqrd_one ** 2
    return (
        2
        * k1
        * rho
        * lam_e_sqrd
        * (lam_e_sqrd_one + (3 * lam_e_sqrd - 1 + 4 * k2 * lam_e_sqrd * lam_e_sqrd_one_sqrd))
        * np.exp(k2 * lam_e_sqrd_one ** 2)
    )


def neo_hooke_cauchy_stress_cir(lambda_phi, lambda_r, mu, rho=1.0):
    return rho * mu * lambda_phi ** 2 * (1 - 1 / (lambda_phi ** 4 * lambda_r ** 2))


def neo_hooke_elastic_modulus_cir(lambda_phi, lambda_r, mu, rho=1.0):
    return 2 * rho * mu * lambda_phi ** 2 * (1 + 1 / (lambda_phi ** 4 * lambda_r ** 2))


class UniformCircumferentialGrowthParams:
    """ Collection of a full set of parameters for the Uniform Circumferential Growth Model.

    Attributes:

    """

    def __init__(self, primary=True, homeostatic=True, **kwargs):
        # initial radius
        self.R0 = kwargs.get("R0", R0)

        # mean blood pressure in Pascal (see [1] Section 6.2 p.219)
        self.mean_pressure = kwargs.get("mean_pressure", MEAN_PRESSURE)

        # time of initial damage (in the past: T0 > 0)
        self.t0 = kwargs.get("t0", T0)

        # initial damage (engineering strain at t=t0)
        self.dr0 = kwargs.get("dr0", DR0)

        # mass fractions
        self.phi_el = kwargs.get("phi_el", PHI_EL)  # elastin
        self.phi_sm = kwargs.get("phi_sm", PHI_SM)  # smooth muscle
        self.phi_co = kwargs.get("phi_co", PHI_CO)  # collagen

        # density
        self.rho = kwargs.get("rho", RHO)

        # half life of collagen
        self.tau = kwargs.get("tau", TAU)
        # growth parameter of collagen
        self.k_sigma = kwargs.get("k_sigma", KSIGMA)

        # material parameters of collagen
        self.k1_co = kwargs.get("k1_co", K1_CO)  # Fung parameter 1 collagen
        self.k2_co = kwargs.get("k2_co", K2_CO)  # Fung parameter 2 collagen

        # material parameters of smooth muscle
        self.k1_sm = kwargs.get("k1_sm", K1_SM)  # Fung parameter 1 smooth muscle
        self.k2_sm = kwargs.get("k2_sm", K2_SM)  # Fung parameter 2 smooth muscle

        # material parameters of elastin matrix
        self.mu_el = kwargs.get("mu_el", MU_EL)  # (3D) Neo-Hooke shear parameter

        # prestretch compared to natural configuration
        self.lam_pre_co = kwargs.get("lam_pre_co", LAM_PRE_CO)  # collagen
        self.lam_pre_sm = kwargs.get("lam_pre_sm", LAM_PRE_SM)  # smooth muscle
        self.lam_pre_el_cir = kwargs.get(
            "lam_pre_el_cir", LAM_PRE_EL_CIR
        )  # elastin circumferential
        self.lam_pre_el_ax = kwargs.get("lam_pre_el_ax", LAM_PRE_EL_AX)  # elastin axial

        if primary:
            # homeostatic collagen stress
            self.sigma_h_co = fung_cauchy_stress(
                self.lam_pre_co, k1=self.k1_co, k2=self.k2_co, rho=self.rho
            )
            # homeostatic smooth muscle stress
            self.sigma_h_sm = fung_cauchy_stress(
                self.lam_pre_sm, k1=self.k1_sm, k2=self.k2_sm, rho=self.rho
            )
            # circumferential stress of elastin in initial configuration
            self.sigma_cir_el = neo_hooke_cauchy_stress_cir(
                self.lam_pre_el_cir, self.lam_pre_el_ax, self.mu_el, rho=self.rho
            )

            # NOTE: prestretch G and elastic modulus C codepend via other material parameters
            # elastic modulus
            self.C_co = kwargs.get("C_co", C_CO)  # collagen
            self.C_el = kwargs.get("C_el", C_EL)  # elastin
            self.C_sm = kwargs.get("C_sm", C_SM)  # smooth muscle
        else:
            # homeostatic collagen stress
            self.sigma_h_co = kwargs.get("sigma_h_co", SIGMA_H_CO)
            # homeostatic smooth muscle stress
            self.sigma_h_sm = kwargs.get("sigma_h_sm", SIGMA_H_SM)
            # circumferential stress of elastin in initial configuration
            self.sigma_cir_el = kwargs.get("simga_cir_el", SIGMA_CIR_EL)

            # NOTE: prestretch G and elastic modulus C codepend via other material parameters
            # elastic modulus
            self.C_co = kwargs.get("C_co", C_CO)  # collagen
            self.C_el = kwargs.get("C_el", C_EL)  # elastin
            self.C_sm = kwargs.get("C_sm", C_SM)  # smooth muscle

        # derived variables
        # material density per unit area
        self.M_e = self.phi_el * self.rho * self.H  # elastin
        self.M_m = self.phi_sm * self.rho * self.H  # smooth muscle
        self.M_c = self.phi_co * self.rho * self.H  # collagen
        # vector of mass fractions
        self.Phi = np.array([self.phi_el, self.phi_sm, self.phi_co])
        # vector of material densities per unit surface area
        self.M = np.array([self.M_e, self.M_m, self.M_c])
        # vector of prestretches
        self.G = np.array([self.lam_pre_el_cir, self.lam_pre_sm, self.lam_pre_co])
        # vector of homeostatic (circumferential) stresses
        # i.e. circumferential stres in initial configuration
        # note that we consider only one collagen fiber family
        self.Sigma_cir = np.array([self.sigma_cir_el, self.sigma_h_sm, self.sigma_h_co])
        # vector of elastic modulus
        self.C = np.array([self.C_el, self.C_sm, self.C_co])
        # initial thickness of aorta wall
        if homeostatic:
            # Laplace Law
            self.H = self.mean_pressure * self.R0 / (self.Phi.dot(self.Sigma_cir))
        else:
            self.H = kwargs.get("H", H)


def stab_margin(tau, k_sigma, sigma_h_c, C=C, M=M, sigma_cir_e=SIGMA_CIR_EL, sigma_h_m=sigma_h_m):
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


def delta_radius(t, tau, m_gnr, dR0, t0):
    """
    Return engineering strain of radius de_r at time t.

    see eq. (3) with (78) + (79) in [1]
    """

    de_r = (1 + (tau * m_gnr - 1) * np.exp(-m_gnr * (t + t0))) * dR0 / (tau * m_gnr)
    return np.squeeze(de_r)


def radius(tau, k_sigma, t, sigma_h_c, dR0, t0):
    """ Return current radius at time t. """
    m_gnr = stab_margin(
        tau, k_sigma, sigma_h_c, C=C, M=M, sigma_cir_e=SIGMA_CIR_EL, sigma_h_m=sigma_h_m
    )
    r = R0 * (1 + delta_radius(t, tau, m_gnr, dR0, t0))
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


if __name__ == "__main__":
    sigma_h_co = fung_cauchy_stress(LAM_PRE_CO, K1_CO, K2_CO, RHO)
    print(f"prestress collagen={sigma_h_co}")
    C_co = fung_elastic_modulus(LAM_PRE_CO, K1_CO, K2_CO, RHO)
    print(f"elastic modulus collagen={C_co}")
    sigma_h_sm = fung_cauchy_stress(LAM_PRE_SM, K1_SM, K2_SM, RHO)
    print(f"prestress smooth muscle={sigma_h_sm}")
    C_sm = fung_elastic_modulus(LAM_PRE_SM, K1_SM, K2_SM, RHO)
    print(f"elastic modulus smooth muscle={C_sm}")
    sigma_h_el_cir = neo_hooke_cauchy_stress_cir(LAM_PRE_EL_CIR, LAM_PRE_EL_AX, MU_EL, RHO)
    print(f"prestress elastin={sigma_h_el_cir}")
    C_el = neo_hooke_elastic_modulus_cir(LAM_PRE_EL_CIR, LAM_PRE_EL_AX, MU_EL, RHO)
    print(f"elastic modulus elastin={C_el}")
