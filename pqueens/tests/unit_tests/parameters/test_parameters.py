"""Test-module for Parameters class."""

import numpy as np
import pytest

import pqueens.parameters.parameters as parameters_module


@pytest.fixture(scope='module')
def parameters_set_1():
    """Parameters dict without random field."""
    parameters_dict = {
        "parameters": {
            "random_variables": {
                "x1": {
                    "dimension": 1,
                    "type": "FLOAT",
                    "distribution": "uniform",
                    "lower_bound": -5,
                    "upper_bound": 10,
                },
                "x2": {
                    "dimension": 2,
                    "distribution": "normal",
                    "mean": [0, 1],
                    "covariance": np.diag([1, 2]),
                },
            }
        }
    }
    return parameters_dict


@pytest.fixture(scope='module')
def parameters_set_2():
    """Parameters dict without random field."""
    parameters_dict = {
        "parameters": {
            "random_variables": {
                "x1": {
                    "dimension": 1,
                    "type": "FLOAT",
                    "distribution": "uniform",
                    "lower_bound": -5,
                    "upper_bound": 10,
                },
                "x2": {
                    "dimension": 1,
                    "distribution": "normal",
                    "mean": [0],
                    "covariance": np.diag([1]),
                },
            }
        }
    }
    return parameters_dict


def create_parameters_singleton(options, pre_processor=None):
    """Create parameters singleton."""
    parameters_module.from_config_create_parameters(options, pre_processor)
    return parameters_module.parameters


@pytest.mark.unit_tests
def test_from_config_create_parameters(parameters_set_1):
    """Test from_config_create_parameters method."""
    parameters = create_parameters_singleton(parameters_set_1)
    rv_x1 = parameters.dict['x1']
    rv_x2 = parameters.dict['x2']

    assert parameters.num_parameters == 3
    assert parameters.parameters_keys == ['x1', 'x2_0', 'x2_1']
    assert parameters.random_field_flag is False
    assert parameters.names == ['x1', 'x2']
    assert rv_x1.lower_bound == -5
    assert rv_x1.upper_bound == 10
    assert rv_x1.size == 1
    assert rv_x1.type == "FLOAT"
    assert rv_x2.lower_bound is None
    assert rv_x2.upper_bound is None
    assert rv_x2.size == 2
    assert rv_x2.type is None


@pytest.mark.unit_tests
def test_draw_samples(parameters_set_1):
    """Test draw_samples method."""
    parameters = create_parameters_singleton(parameters_set_1)
    np.random.seed(41)
    samples = parameters.draw_samples(1)
    np.testing.assert_almost_equal(samples, np.array([[-1.23615, 0.11724, 0.57436]]), decimal=5)
    samples = parameters.draw_samples(2)
    np.testing.assert_almost_equal(
        samples, np.array([[-4.34796, -1.23961, 1.84974], [-3.25364, 0.41658, 1.34302]]), decimal=5
    )
    samples = parameters.draw_samples(1000)
    mean = np.mean(samples, axis=0)
    variance = np.var(samples, axis=0)
    np.testing.assert_almost_equal(mean, np.array([2.42511, 0.02864, 1.03762]), decimal=5)
    np.testing.assert_almost_equal(variance, np.array([19.00948, 1.03104, 2.09257]), decimal=5)


@pytest.mark.unit_tests
def test_joint_logpdf(parameters_set_1):
    """Test joint_logpdf method."""
    parameters = create_parameters_singleton(parameters_set_1)
    samples = np.array([1, 2, 3])
    logpdf = parameters.joint_logpdf(samples)
    np.testing.assert_almost_equal(logpdf, np.array([-7.89250]), decimal=5)
    samples = np.array([[20, 2, 3], [-2, 4, -2]])
    logpdf = parameters.joint_logpdf(samples)
    np.testing.assert_almost_equal(logpdf, np.array([-np.inf, -15.14250]), decimal=5)


@pytest.mark.unit_tests
def test_inverse_cdf_transform(parameters_set_1, parameters_set_2):
    """Test inverse_cdf_transform method."""
    parameters = create_parameters_singleton(parameters_set_1)
    samples = np.array([0.5, 0.1, 0.6])
    with pytest.raises(ValueError):
        parameters.inverse_cdf_transform(samples)

    parameters = create_parameters_singleton(parameters_set_2)
    samples = np.array([0.5, 0.1])
    transformed_samples = parameters.inverse_cdf_transform(samples)
    np.testing.assert_almost_equal(transformed_samples, np.array([[2.5, -1.28155]]), decimal=5)

    samples = np.array([[0.5, 0.1], [1.0, 0.1]])
    transformed_samples = parameters.inverse_cdf_transform(samples)
    np.testing.assert_almost_equal(
        transformed_samples, np.array([[2.50000, -1.28155], [10.00000, -1.28155]]), decimal=5
    )


@pytest.mark.unit_tests
def test_sample_as_dict(parameters_set_1):
    """Test sample_as_dict method."""
    parameters = create_parameters_singleton(parameters_set_1)
    sample = np.array([0.5, 0.1, 0.6])
    sample_dict = parameters.sample_as_dict(sample)
    assert sample_dict == {'x1': 0.5, 'x2_0': 0.1, 'x2_1': 0.6}


@pytest.mark.unit_tests
def test_to_list(parameters_set_1):
    """Test to_list method."""
    parameters = create_parameters_singleton(parameters_set_1)
    parameters_list = parameters.to_list()
    assert isinstance(parameters_list, list)
    assert len(parameters_list) == 2


# -------------------------------------------------------------------------------
# -------------------------   With random field   -------------------------------
# -------------------------------------------------------------------------------
@pytest.fixture(scope='module')
def parameters_set_3():
    """Parameters dict with random field."""
    parameters_dict = {
        "parameters": {
            "random_variables": {
                "x1": {
                    "dimension": 1,
                    "type": "FLOAT",
                    "distribution": "uniform",
                    "lower_bound": -5,
                    "upper_bound": 10,
                },
                "x2": {
                    "dimension": 2,
                    "distribution": "normal",
                    "mean": [0, 1],
                    "covariance": np.diag([1, 2]),
                },
            },
            "random_fields": {
                "random_inflow": {
                    "corr_length": 1.0,
                    "std_hyperparam_rf": 0.001,
                    "mean_type": "constant",
                    "mean_param": 0,
                }
            },
        }
    }
    return parameters_dict


@pytest.fixture(scope='module')
def pre_processor():
    """Create basic preprocessor class instance."""

    class PreProcessor:
        """Basic preprocessor class."""

        def __init__(self):
            """Initialize."""
            self.coords_dict = {
                'random_inflow': {
                    'keys': ['random_inflow_0', 'random_inflow_1', 'random_inflow_2'],
                    'coords': [0.0, 0.5, 1.0],
                }
            }

    return PreProcessor()


@pytest.mark.unit_tests
def test_from_config_create_parameters(parameters_set_3, pre_processor):
    """Test from_config_create_parameters method with random fields."""
    parameters = create_parameters_singleton(parameters_set_3, pre_processor)

    assert parameters.num_parameters == 5
    assert parameters.parameters_keys == [
        'x1',
        'x2_0',
        'x2_1',
        'random_inflow_0',
        'random_inflow_1',
        'random_inflow_2',
    ]
    assert parameters.random_field_flag is True
    assert parameters.names == ['x1', 'x2', 'random_inflow']