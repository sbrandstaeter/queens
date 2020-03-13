# pylint: disable=line-too-long
import pqueens.example_simulator_functions.uniform_growth_1D_stab_margin as uniform_growth_1D_stab_margin
import pqueens.example_simulator_functions.uniform_circumferential_growth_and_remodelling as uniform_growth_1D
import pqueens.example_simulator_functions.uniform_growth_1D_lsq as uniform_growth_1D_lsq

# pylint: enable=line-too-long

TAU = uniform_growth_1D_lsq.TAU
DRO = uniform_growth_1D_lsq.DR0
T0 = uniform_growth_1D_lsq.T0

# point of time to evaluate the delta in radius
t = uniform_growth_1D_lsq.T_END


def main(job_id, params):
    """
    Interface to displacement of radius of GnR model.

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

    tau = params.get("tau", TAU)
    # make sure that tau for m_gnr and for delta_radius is equal
    params["tau"] = tau
    dR0 = params.get("dR0", DRO)
    t0 = params.get("t0", T0)
    m_gnr = uniform_growth_1D_stab_margin.main(job_id, params)

    dR = uniform_growth_1D.delta_radius(t, tau, m_gnr, dR0, t0)

    return dR


if __name__ == "__main__":
    empty_dict = dict()
    print(main(0, empty_dict))
