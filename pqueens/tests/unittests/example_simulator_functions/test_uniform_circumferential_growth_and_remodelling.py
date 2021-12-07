"""Test module for uniform circumferential growth and remodelling model."""

import numpy as np
import pytest

# pylint: disable=line-too-long
import pqueens.tests.integration_tests.example_simulator_functions.uniform_circumferential_growth_and_remodelling as uni_cir_gnr
import pqueens.tests.integration_tests.example_simulator_functions.uniform_circumferential_growth_and_remodelling_radial_displacement as uni_cir_gnr_dr
import pqueens.tests.integration_tests.example_simulator_functions.uniform_circumferential_growth_and_remodelling_radial_displacement_logpdf as uni_cir_gnr_logpdf
import pqueens.tests.integration_tests.example_simulator_functions.uniform_circumferential_growth_and_remodelling_radial_displacement_lsq as uni_cir_gnr_lsq
import pqueens.tests.integration_tests.example_simulator_functions.uniform_circumferential_growth_and_remodelling_radial_strain as uni_cir_gnr_de_r
import pqueens.tests.integration_tests.example_simulator_functions.uniform_circumferential_growth_and_remodelling_stability_margin as uni_cir_gnr_mgnr
import pqueens.tests.integration_tests.example_simulator_functions.uniform_circumferential_growth_and_remodelling_thickness as uni_cir_gnr_thickness

# pylint: enable=line-too-long


@pytest.fixture(scope='module', params=[True, False])
def parameter_set(request):
    """Switch for primary vs base parameter sets.

    Returns "True" for primary parameter and "False" for base parameter
    set.
    """
    return request.param


@pytest.fixture(scope='module')
def base_param_names():
    """Return list of possible names of parameter."""
    params_names = {
        "C_co",
        "C_el",
        "C_sm",
        "de_r0",
        "k_sigma",
        "mean_pressure",
        "phi_co",
        "phi_el",
        "phi_sm",
        "r0",
        "rho",
        "sigma_cir_el",
        "sigma_h_co",
        "sigma_h_sm",
        "t0",
        "tau",
    }

    derived_params = {"C", "h", "M", "M_co", "M_el", "M_sm", "Phi", "Sigma_cir"}
    return [params_names, derived_params]


@pytest.fixture(scope='module')
def primary_param_names():
    """Return possible name of parameter."""
    params_names = {
        "de_r0",
        "k1_co",
        "k1_sm",
        "k2_co",
        "k2_sm",
        "k_sigma",
        "lam_pre_ax_el",
        "lam_pre_cir_el",
        "lam_pre_co",
        "lam_pre_sm",
        "mean_pressure",
        "mu_el",
        "phi_co",
        "phi_el",
        "phi_sm",
        "r0",
        "rho",
        "t0",
        "tau",
    }

    derived_params = {
        "C",
        "C_co",
        "C_el",
        "C_sm",
        "G",
        "M",
        "M_co",
        "M_el",
        "M_sm",
        "Phi",
        "Sigma_cir",
        "h",
        "sigma_cir_el",
        "sigma_h_co",
        "sigma_h_sm",
    }
    return [params_names, derived_params]


@pytest.fixture(scope='module')
def default_primary_param_dict(primary_param_names):
    # create a parameter dictionary
    primary_params, derived_params = primary_param_names
    param_dict = dict()
    for param_name in primary_params:
        value = getattr(uni_cir_gnr, param_name.upper())
        param_dict.update({param_name: value})
    return param_dict


@pytest.fixture(scope="module")
def stability_margin_goal_value():
    """Return the stability margi's goal value for default parameter set."""
    m_gnr_goal = -4.5960674991981504e-4
    return m_gnr_goal


@pytest.fixture(scope='module')
def matlab_data():
    """The test data is taken from Christian Cyron's Matlab Code with the same
    parameters."""
    # timestamps to compare at
    t = np.array([0, 720, 1440, 2160, 2900])
    # initial radius
    r0 = 1.25e-2
    # engineering strain from matlab
    de_matlab = np.array(
        [
            0.002477004816501293,
            0.03364802630268014,
            0.0770456950705312,
            0.13746584059958264,
            0.2243426396272093,
        ]
    )
    # radial displacements
    dr_matlab = de_matlab * r0
    # radius
    r_matlab = dr_matlab + r0
    data = dict()
    data["t"] = t
    data["r0"] = r0
    data["de_matlab"] = de_matlab
    data["dr_matlab"] = dr_matlab
    data["r_matlab"] = r_matlab
    return data


@pytest.mark.unit_tests
def test_default_base_params(base_param_names):
    """Test that the default base parameter set is set correctly."""
    param_names = set()
    # create a complete list of param names (i.e., base and derived params)
    for names in base_param_names:
        for name in names:
            param_names.add(name)
    # do not supply any optional keyword args that would contain the parameters -> use default
    gnr_params = uni_cir_gnr.UniformCircumferentialGrowthAndRemodellingParams(primary=False)
    for name in param_names:
        param_goal = getattr(uni_cir_gnr, name.upper())
        param_is = getattr(gnr_params, name)
        assert np.allclose(param_goal, param_is)


@pytest.mark.unit_tests
def test_default_primary_params(primary_param_names):
    """Test that the default primary parameter set is set correctly."""
    param_names = set()
    # create a complete list of param names (i.e., primary and derived params)
    for names in primary_param_names:
        for name in names:
            param_names.add(name)
    # do not supply any optional keyword args that would contain the parameters -> use default
    gnr_params = uni_cir_gnr.UniformCircumferentialGrowthAndRemodellingParams(primary=True)
    for name in param_names:
        param_goal = getattr(uni_cir_gnr, name.upper())
        param_is = getattr(gnr_params, name)
        assert np.allclose(param_goal, param_is)


@pytest.mark.unit_tests
def test_default_stability_margin(parameter_set, stability_margin_goal_value):
    """Test that the default stability margin is calculated correctly."""
    # value of stability margin with Christian Cyron's Matlab Code with default parameters
    gnr_params = uni_cir_gnr.UniformCircumferentialGrowthAndRemodellingParams(primary=parameter_set)
    assert np.isclose(stability_margin_goal_value, gnr_params.m_gnr)


@pytest.mark.unit_tests
def test_set_base_params(base_param_names):
    """Test that the default primary parameter set is set correctly."""

    value_goal = 3.14159
    base_params, derived_params = base_param_names
    for param_name in base_params:
        param_dict = {param_name: value_goal}
        gnr_params = uni_cir_gnr.UniformCircumferentialGrowthAndRemodellingParams(
            primary=False, **param_dict
        )
        value_is = getattr(gnr_params, param_name)
        assert np.allclose(value_goal, value_is)


@pytest.mark.unit_tests
def test_set_primary_params(primary_param_names):
    """Test that the default primary parameter set is set correctly."""

    value_goal = 0.14159
    primary_params, derived_params = primary_param_names
    for param_name in primary_params:
        param_dict = {param_name: value_goal}
        gnr_params = uni_cir_gnr.UniformCircumferentialGrowthAndRemodellingParams(
            primary=True, **param_dict
        )
        value_is = getattr(gnr_params, param_name)
        assert np.allclose(value_goal, value_is)


@pytest.mark.unit_tests
def test_warn_unused_param():
    """Test that an unused parameter raises a warning."""

    param_dict = {"my_parameter": 3.14159}
    with pytest.raises(Warning):
        uni_cir_gnr.UniformCircumferentialGrowthAndRemodellingParams(primary=True, **param_dict)


@pytest.mark.unit_tests
def test_strains_default_params(matlab_data, parameter_set):
    """Test dynamics of engineering strains of radius with default
    parameters."""
    t = matlab_data["t"]
    de_matlab = matlab_data["de_matlab"]

    gnr_model = uni_cir_gnr.UniformCircumferentialGrowthAndRemodelling(primary=parameter_set)

    de = gnr_model.de_r(t)

    assert np.allclose(de_matlab, de, atol=1e-17)


@pytest.mark.unit_tests
def test_radial_displacements_default_params(matlab_data, parameter_set):
    """Test dynamics of radial displacements with default parameters."""
    t = matlab_data["t"]
    dr_matlab = matlab_data["dr_matlab"]

    gnr_model = uni_cir_gnr.UniformCircumferentialGrowthAndRemodelling(primary=parameter_set)

    dr = gnr_model.dr(t)

    assert np.allclose(dr_matlab, dr, atol=1e-17)


@pytest.mark.unit_tests
def test_radius_dynamics_default_primary_params(matlab_data, parameter_set):
    """Test dynamics of engineering strain with default parameters."""
    t = matlab_data["t"]
    r_matlab = matlab_data["r_matlab"]

    gnr_model = uni_cir_gnr.UniformCircumferentialGrowthAndRemodelling(primary=parameter_set)

    r = gnr_model.radius(t)

    assert np.allclose(r_matlab, r, atol=1e-17)


@pytest.mark.unit_tests
def test_radial_displacement_model(default_primary_param_dict, matlab_data):
    """Test radial displacement model."""
    dr_matlab = matlab_data["dr_matlab"]

    param_dict = default_primary_param_dict.copy()
    param_dict["t"] = matlab_data["t"]

    dummy_job_id = 0
    dr = uni_cir_gnr_dr.main(dummy_job_id, param_dict)

    assert np.allclose(dr_matlab, dr, atol=1e-17)


@pytest.mark.unit_tests
def test_radial_displacement_logpdf_model(default_primary_param_dict):
    """Test radial displacement gaussian logpdf model based in least squares
    model."""
    logpdf_goal = 0.8224330526609611

    param_dict = default_primary_param_dict.copy()

    dummy_job_id = 0
    logpdf = uni_cir_gnr_logpdf.main(dummy_job_id, param_dict)

    assert np.isclose(logpdf_goal, logpdf, atol=1e-17)


@pytest.mark.unit_tests
def test_radial_displacement_lsq_model(default_primary_param_dict):
    """Test radial displacement least squares model."""
    lsq_goal = 1.0332767386233007e-05

    param_dict = default_primary_param_dict.copy()

    dummy_job_id = 0
    lsq = uni_cir_gnr_lsq.main(dummy_job_id, param_dict)

    assert np.isclose(lsq_goal, lsq, atol=1e-17)


@pytest.mark.unit_tests
def test_radial_strain_model(default_primary_param_dict, matlab_data):
    """Test radial strain model."""
    de_matlab = matlab_data["de_matlab"]

    param_dict = default_primary_param_dict.copy()
    param_dict["t"] = matlab_data["t"]

    dummy_job_id = 0
    de = uni_cir_gnr_de_r.main(dummy_job_id, param_dict)

    assert np.allclose(de_matlab, de, atol=1e-17)


@pytest.mark.unit_tests
def test_stability_margin_model(default_primary_param_dict, stability_margin_goal_value):
    """Test stability margin model."""

    dummy_job_id = 0
    m_gnr = uni_cir_gnr_mgnr.main(dummy_job_id, default_primary_param_dict)

    assert np.isclose(stability_margin_goal_value, m_gnr, atol=1e-17)


@pytest.mark.unit_tests
def test_thickness_model(default_primary_param_dict):
    """Test thickness model."""
    h_goal = uni_cir_gnr.H

    dummy_job_id = 0
    h = uni_cir_gnr_thickness.main(dummy_job_id, default_primary_param_dict)

    assert np.isclose(h_goal, h, atol=1e-17)
