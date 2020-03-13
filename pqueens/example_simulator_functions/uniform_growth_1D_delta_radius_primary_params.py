# pylint: disable=line-too-long
import pqueens.example_simulator_functions.uniform_growth_1D_stab_margin_primary_params as uniform_growth_1D_stab_margin_primary_params
import pqueens.example_simulator_functions.uniform_circumferential_growth_and_remodelling as uniform_growth_1D
import pqueens.example_simulator_functions.uniform_growth_1D_lsq as uniform_growth_1D_lsq

# pylint: enable=line-too-long

TAU = uniform_growth_1D_lsq.TAU
DRO = uniform_growth_1D_lsq.DR0
T0 = uniform_growth_1D_lsq.T0

# point of time to evaluate the delta in radius
T = uniform_growth_1D_lsq.T_END


def main(job_id, params):
    """
    Interface to engineering strain of radius of GnR model parameterized with primary parameters.

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

    # current time to evaluate growth at
    t = params.get("t", T)

    tau = params.get("tau", TAU)
    # make sure that tau for m_gnr and for delta_radius is equal
    params["tau"] = tau
    dR0 = params.get("dR0", DRO)
    # time of perturbation, i.e. initiation of growth
    t0 = params.get("t0", T0)

    # stability margin
    m_gnr = uniform_growth_1D_stab_margin_primary_params.main(job_id, params)

    de_r = uniform_growth_1D.delta_radius(t, tau, m_gnr, dR0, t0)

    return de_r


if __name__ == "__main__":
    empty_dict = dict()
    print(main(0, empty_dict))
