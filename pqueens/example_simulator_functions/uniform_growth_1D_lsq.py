import numpy as np

from pqueens.example_simulator_functions.uniform_growth_1D import delta_radius

days_to_seconds = 1  # 86400
years_to_days = 365.25
years_to_seconds = years_to_days * days_to_seconds

# real parameter values
tau = 100 * days_to_seconds
# k_sigma = 0.001107733872774 / days_to_seconds
k_sigma = 4.933441e-04 / days_to_seconds
sigma_h_c = 200000
dR0 = 0.002418963148596
t0 = 2 * years_to_seconds


t_end = 3000 * days_to_seconds

num_meas = 8

# t_meas = np.linspace(-t0, t_end, num_meas)
t_meas = np.linspace(0.0, t_end, num_meas)
# t_meas = [0, 730, 1460, 2190]
dr_meas = delta_radius(t_meas, tau, k_sigma, sigma_h_c, dR0, t0)

rng_state = np.random.get_state()
seed = 24
np.random.seed(seed)
std_noise = 1.0e-3  # 1.0e-2
noise = np.random.normal(0, std_noise, num_meas)

np.random.set_state(rng_state)

dr_meas = dr_meas + noise


def main(job_id, params):
    """ Interface to least squares of GnR model

    UNITS:
    - length [m]
    - mass [kg]
    - stress [Pa]

    Args:
        job_id (int):   ID of job
        params (dict):  Dictionary with parameters

    Returns:
        float:          Value of GnR model at parameters
                        specified in input dict
    """

    dr_sim = delta_radius(
        t_meas, params['tau'], params['k_sigma'], params['sigma_h_c'], params['dR0'], params['t0']
    )

    residuals = dr_meas - dr_sim

    return residuals
