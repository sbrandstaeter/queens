"""Unit tests for Bayesian multi-fidelity inverse analysis iterator."""

from unittest.mock import patch

import numpy as np
import pytest

import pqueens.parameters.parameters as parameters_module
from pqueens.iterators.bmfia_iterator import BMFIAIterator
from pqueens.models.simulation_model import SimulationModel


# ------------ fixtures and params -----------------------------------
@pytest.fixture()
def result_description():
    """Fixture for a dummy result description."""
    description = {"write_results": True}
    return description


@pytest.fixture()
def global_settings():
    """Fixture for dummy global settings."""
    global_set = {'output_dir': 'dummyoutput', 'experiment_name': 'dummy_exp_name'}
    return global_set


@pytest.fixture()
def parameters():
    """Fixture for dummy parameters."""
    rv = {"type": "random_variable", "distribution": "uniform", "lower_bound": -2, "upper_bound": 2}
    params = {"x1": rv, "x2": rv}
    parameters_module.from_config_create_parameters({'parameters': params})
    return params


@pytest.fixture()
def dummy_model(parameters):
    """Fixture for dummy model."""
    model_name = 'dummy'
    interface = 'my_dummy_interface'
    model = SimulationModel(model_name, interface)
    return model


@pytest.fixture()
def default_bmfia_iterator(result_description, global_settings, dummy_model):
    """Dummy iterator for testing."""
    result_description = result_description
    global_settings = global_settings
    features_config = 'no_features'
    hf_model = dummy_model
    lf_model = dummy_model
    output_label = ['y']
    coord_labels = ['x_1', 'x_2']
    settings_probab_mapping = {'features_config': 'no_features'}
    db = 'dummy_db'
    x_train = np.array([[1, 2], [3, 4]])
    Y_LF_train = np.array([[2], [3]])
    Y_HF_train = np.array([[2.2], [3.3]])
    Z_train = np.array([[4], [5]])
    coords_experimental_data = np.array([[1, 2], [3, 4]])
    time_vec = np.array([1, 3])
    y_obs_vec = np.array([[2.1], [3.1]])

    iterator = BMFIAIterator(
        result_description,
        global_settings,
        features_config,
        hf_model,
        lf_model,
        output_label,
        coord_labels,
        settings_probab_mapping,
        db,
        x_train,
        Y_LF_train,
        Y_HF_train,
        Z_train,
        coords_experimental_data,
        time_vec,
        y_obs_vec,
    )

    return iterator


@pytest.fixture()
def settings_probab_mapping(config, approximation_name):
    """Dummy settings for the probabilistic mapping for testing."""
    settings = config[approximation_name]
    return settings


@pytest.fixture()
def approximation_name():
    """Dummy approximation name for testing."""
    name = 'joint_density_approx'
    return name


@pytest.fixture()
def config():
    """Fixture for dummy configuration."""
    config = {
        "joint_density_approx": {
            "type": "gp_approximation_gpy",
            "features_config": "opt_features",
            "num_features": 1,
            "X_cols": 1,
        }
    }
    return config


def my_mock_design(*args):
    """Implementation of mock design method."""
    x_train = np.array([[1, 1]])
    return x_train, args


# -------------- Actual tests -------------------------------------
def test_init(result_description, global_settings, dummy_model, settings_probab_mapping):
    """Test the init of the Bayesian multi-fidelity iterator."""
    features_config = 'no_features'
    hf_model = dummy_model
    lf_model = dummy_model
    output_label = 'y'
    coord_labels = ['x1', 'x2', 'x3']
    settings_probab_mapping = settings_probab_mapping
    db = 'dummy'
    x_train = np.array([[1, 1, 1], [2, 2, 2]])
    Y_LF_train = np.array([[1, 2, 3], [4, 5, 6]])
    Y_HF_train = np.array([[1.1, 2.1, 3.1], [4.1, 5.1, 6.1]])
    Z_train = np.array([[7, 7, 7], [8, 8, 8]])
    coords_experimental_data = np.array([[2, 2, 2], [9, 9, 9]])
    time_vec = np.linspace(1, 10, 3)
    y_obs_vec = np.array([[2.2, 3.2, 4.2], [5.2, 6.2, 7.2]])

    iterator = BMFIAIterator(
        result_description,
        global_settings,
        features_config,
        hf_model,
        lf_model,
        output_label,
        coord_labels,
        settings_probab_mapping,
        db,
        x_train,
        Y_LF_train,
        Y_HF_train,
        Z_train,
        coords_experimental_data,
        time_vec,
        y_obs_vec,
    )

    # ---- tests / asserts -------------------------
    assert iterator.result_description == result_description
    np.testing.assert_array_equal(iterator.X_train, x_train)
    np.testing.assert_array_equal(iterator.Y_LF_train, Y_LF_train)
    np.testing.assert_array_equal(iterator.Y_HF_train, Y_HF_train)
    np.testing.assert_array_equal(iterator.Z_train, Z_train)
    assert iterator.features_config == features_config
    assert iterator.hf_model == hf_model
    assert iterator.lf_model == lf_model
    np.testing.assert_array_equal(iterator.coords_experimental_data, coords_experimental_data)
    np.testing.assert_array_equal(iterator.time_vec, time_vec)
    assert iterator.output_label == output_label
    assert iterator.coord_labels == coord_labels
    np.testing.assert_array_equal(iterator.y_obs_vec, y_obs_vec)
    assert iterator.settings_probab_mapping == settings_probab_mapping
    assert iterator.db == db


def test_calculate_optimal_x_train(dummy_model, mocker):
    """Test calculation of optimal *x_train*.

    **Note:** Here we return the input arguments of the design method to
    later be able to test if the arguments were correct.
    """
    expected_x_train = np.array([[1, 1]])  # return of mock_design
    model = dummy_model
    initial_design_dict = {'test': 'test'}
    mo_1 = mocker.patch(
        'pqueens.iterators.bmfia_iterator.BMFIAIterator._get_design_method',
        return_value=my_mock_design,
    )

    x_train, (arg0, arg1) = BMFIAIterator._calculate_optimal_x_train(initial_design_dict, model)

    np.testing.assert_array_almost_equal(x_train, expected_x_train)
    assert mo_1.call_args[0][0] == initial_design_dict

    # test if the input arguments are correct
    assert arg0 == initial_design_dict
    assert arg1 == dummy_model


def test_get_design_method(mocker):
    """Test the selection of the design method."""
    # test the random design
    initial_design_dict = {"type": "random"}
    mo_1 = mocker.patch(
        'pqueens.iterators.bmfia_iterator.BMFIAIterator._random_design',
        return_value=my_mock_design,
    )

    design = BMFIAIterator._get_design_method(initial_design_dict)
    assert design == mo_1

    # test invalid design
    with pytest.raises(NotImplementedError):
        initial_design_dict = {'type': 'randommm'}
        BMFIAIterator._get_design_method(initial_design_dict)

    # test invalid key in dictionary
    with pytest.raises(AssertionError):
        initial_design_dict = {'typeeee': 'random'}
        BMFIAIterator._get_design_method(initial_design_dict)

    # test invalid data type of input
    with pytest.raises(AssertionError):
        initial_design_dict = 1
        BMFIAIterator._get_design_method(initial_design_dict)


def test_random_design(dummy_model, parameters):
    """Test for the uniformly random design method."""
    initial_design_dict = {"seed": 1, "num_HF_eval": 1}
    x_train = np.array([[-0.33191198, 0.881297]])
    x_out = BMFIAIterator._random_design(initial_design_dict, dummy_model)

    np.testing.assert_array_almost_equal(x_train, x_out, decimal=4)


def test_core_run(default_bmfia_iterator, mocker):
    """Test the core run of the iterator."""
    z_train_in = np.array([[4], [5]])
    y_hf_train_in = np.array([[2.2], [3.3]])

    mo_1 = mocker.patch('pqueens.iterators.bmfia_iterator.BMFIAIterator.eval_model')
    mo_2 = mocker.patch(
        'pqueens.iterators.bmfia_iterator.BMFIAIterator._set_feature_strategy',
        return_value=z_train_in,
    )

    z_train_out, y_hf_train_out = default_bmfia_iterator.core_run()

    # Actual tests / asserts
    mo_1.assert_called_once()
    mo_2.assert_called_once()
    np.testing.assert_array_equal(mo_2.call_args[0][0], default_bmfia_iterator.Y_LF_train)
    np.testing.assert_array_equal(mo_2.call_args[0][1], default_bmfia_iterator.X_train)
    np.testing.assert_array_equal(
        mo_2.call_args[0][2], default_bmfia_iterator.coords_experimental_data
    )
    np.testing.assert_array_equal(z_train_out, z_train_in)
    np.testing.assert_array_equal(y_hf_train_out, y_hf_train_in)


def test_evaluate_LF_model_for_X_train(default_bmfia_iterator):
    """Test evaluation of LF model with test data."""
    with patch.object(
        default_bmfia_iterator.lf_model, 'evaluate', return_value={'mean': np.array([1, 1])}
    ) as mo_1:

        default_bmfia_iterator._evaluate_LF_model_for_X_train()

        mo_1.assert_called_once()
        np.testing.assert_array_equal(np.array([[1, 1]]), default_bmfia_iterator.Y_LF_train)


def test_evaluate_HF_model_for_X_train(default_bmfia_iterator):
    """Test evaluation of HF model with test data."""
    with patch.object(
        default_bmfia_iterator.hf_model, 'evaluate', return_value={'mean': np.array([1, 1])}
    ) as mo_2:

        default_bmfia_iterator._evaluate_HF_model_for_X_train()

        # Actual asserts / tests
        mo_2.assert_called_once()
        np.testing.assert_array_equal(np.array([[1, 1]]), default_bmfia_iterator.Y_HF_train)


def test_set_feature_strategy(default_bmfia_iterator, mocker):
    """Test the generation of low fidelity informative features."""
    # test wrong input dimensions 1) of y_lf_mat
    y_lf_mat = np.array([1, 2, 3])
    x_mat = np.array([[4, 5, 6]])
    coords_mat = np.array([[7, 8, 9]])
    with pytest.raises(AssertionError):
        default_bmfia_iterator._set_feature_strategy(y_lf_mat, x_mat, coords_mat)

    # test wrong input dimensions 2) of y_x_mat
    y_lf_mat = np.array([[1, 2, 3]])
    x_mat = np.array([4, 5, 6])
    coords_mat = np.array([[7, 8, 9]])

    with pytest.raises(AssertionError):
        default_bmfia_iterator._set_feature_strategy(y_lf_mat, x_mat, coords_mat)

    # test wrong input dimensions 3) of coords_mat
    y_lf_mat = np.array([[1, 2, 3]])
    x_mat = np.array([[4, 5, 6]])
    coords_mat = np.array([7, 8, 9])

    with pytest.raises(AssertionError):
        default_bmfia_iterator._set_feature_strategy(y_lf_mat, x_mat, coords_mat)

    # test wrong features_config
    y_lf_mat = np.array([[1, 2, 3]])
    x_mat = np.array([[4, 5, 6]])
    coords_mat = np.array([[7, 8, 9]])
    default_bmfia_iterator.settings_probab_mapping["features_config"] = "dummy"
    with pytest.raises(ValueError):
        default_bmfia_iterator._set_feature_strategy(y_lf_mat, x_mat, coords_mat)

    # test correct settings for all options
    y_lf_mat = np.array([[1, 2, 3]])
    x_mat = np.array([[4, 5, 6]])
    coords_mat = np.array([[7, 8, 9]])
    mo_man = mocker.patch('pqueens.iterators.bmfia_iterator.BMFIAIterator._get_man_features')
    mo_opt = mocker.patch('pqueens.iterators.bmfia_iterator.BMFIAIterator._get_opt_features')
    mo_coord = mocker.patch('pqueens.iterators.bmfia_iterator.BMFIAIterator._get_coord_features')
    mo_no = mocker.patch('pqueens.iterators.bmfia_iterator.BMFIAIterator._get_no_features')
    mo_time = mocker.patch('pqueens.iterators.bmfia_iterator.BMFIAIterator._get_time_features')

    default_bmfia_iterator.settings_probab_mapping["features_config"] = "man_features"
    default_bmfia_iterator._set_feature_strategy(y_lf_mat, x_mat, coords_mat)

    default_bmfia_iterator.settings_probab_mapping["features_config"] = "opt_features"
    default_bmfia_iterator._set_feature_strategy(y_lf_mat, x_mat, coords_mat)

    default_bmfia_iterator.settings_probab_mapping["features_config"] = "coord_features"
    default_bmfia_iterator._set_feature_strategy(y_lf_mat, x_mat, coords_mat)

    default_bmfia_iterator.settings_probab_mapping["features_config"] = "no_features"
    default_bmfia_iterator._set_feature_strategy(y_lf_mat, x_mat, coords_mat)

    default_bmfia_iterator.settings_probab_mapping["features_config"] = "time_features"
    default_bmfia_iterator._set_feature_strategy(y_lf_mat, x_mat, coords_mat)

    mo_man.assert_called_once()
    mo_opt.assert_called_once()
    mo_coord.assert_called_once()
    mo_no.assert_called_once()
    mo_time.assert_called_once()


def test_get_man_features(default_bmfia_iterator):
    """Test generation of manual features."""
    y_lf_mat = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    x_mat = np.array([[4, 5, 6], [4, 5, 6], [4, 5, 6]])
    coords_mat = np.array([[7, 8, 9], [10, 11, 12], [13, 14, 15]])

    # test man_features without specifing 'X_cols' --> KeyError
    default_bmfia_iterator.settings_probab_mapping["features_config"] = "man_features"
    with pytest.raises(KeyError):
        z_mat = default_bmfia_iterator._set_feature_strategy(y_lf_mat, x_mat, coords_mat)

    # test man_features with X_col not in list format
    y_lf_mat = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    x_mat = np.array([[4, 5, 6], [4, 5, 6], [4, 5, 6]])
    coords_mat = np.array([[7, 8, 9], [10, 11, 12], [13, 14, 15]])
    default_bmfia_iterator.settings_probab_mapping["features_config"] = "man_features"
    default_bmfia_iterator.settings_probab_mapping['X_cols'] = 0
    with pytest.raises(AssertionError):
        z_mat = default_bmfia_iterator._set_feature_strategy(y_lf_mat, x_mat, coords_mat)

    # test man_features with X_col as empty list
    y_lf_mat = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    x_mat = np.array([[4, 5, 6], [4, 5, 6], [4, 5, 6]])
    coords_mat = np.array([[7, 8, 9], [10, 11, 12], [13, 14, 15]])
    default_bmfia_iterator.settings_probab_mapping["features_config"] = "man_features"
    default_bmfia_iterator.settings_probab_mapping['X_cols'] = []
    with pytest.raises(AssertionError):
        z_mat = default_bmfia_iterator._set_feature_strategy(y_lf_mat, x_mat, coords_mat)

    # test man_features with correct configuration
    expected_z_mat = np.array(
        [[[1, 2, 3], [1, 2, 3], [1, 2, 3]], [[4, 4, 4], [4, 4, 4], [4, 4, 4]]]
    )
    y_lf_mat = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    x_mat = np.array([[4, 5, 6], [4, 5, 6], [4, 5, 6]])
    coords_mat = np.array([[7, 8, 9], [10, 11, 12], [13, 14, 15]])

    # test man features with correct settings
    default_bmfia_iterator.settings_probab_mapping["features_config"] = "man_features"
    default_bmfia_iterator.settings_probab_mapping['X_cols'] = [0]
    z_mat = default_bmfia_iterator._set_feature_strategy(y_lf_mat, x_mat, coords_mat)
    np.testing.assert_array_almost_equal(z_mat, expected_z_mat, decimal=4)


def test_get_opt_features(default_bmfia_iterator):
    """Test generation of optimal features."""
    y_lf_mat = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    x_mat = np.array([[4, 5, 6], [4, 5, 6], [4, 5, 6]])
    coords_mat = np.array([[7, 8, 9], [10, 11, 12], [13, 14, 15]])

    # test opt_features with num features < 1 --> error
    default_bmfia_iterator.settings_probab_mapping["features_config"] = "opt_features"
    default_bmfia_iterator.settings_probab_mapping["num_features"] = 0
    with pytest.raises(AssertionError):
        default_bmfia_iterator._set_feature_strategy(y_lf_mat, x_mat, coords_mat)

    # test opt_features with num features None --> error
    default_bmfia_iterator.settings_probab_mapping["features_config"] = "opt_features"
    default_bmfia_iterator.settings_probab_mapping["num_features"] = None
    with pytest.raises(AssertionError):
        default_bmfia_iterator._set_feature_strategy(y_lf_mat, x_mat, coords_mat)

    # test opt features with correct set-up
    default_bmfia_iterator.settings_probab_mapping["features_config"] = "opt_features"
    default_bmfia_iterator.settings_probab_mapping["num_features"] = 1
    with pytest.raises(NotImplementedError):
        default_bmfia_iterator._set_feature_strategy(y_lf_mat, x_mat, coords_mat)


def test_get_coord_features(default_bmfia_iterator):
    """Test generation of coordinate features."""
    y_lf_mat = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    x_mat = np.array([[4, 5, 6], [4, 5, 6], [4, 5, 6]])
    coords_mat = np.array([[7, 8, 9], [10, 11, 12], [13, 14, 15]])

    # test coord_features without specifing 'coord_cols' --> KeyError
    default_bmfia_iterator.settings_probab_mapping["features_config"] = "coord_features"
    with pytest.raises(KeyError):
        z_mat = default_bmfia_iterator._set_feature_strategy(y_lf_mat, x_mat, coords_mat)

    # test coord_features with empty col list
    default_bmfia_iterator.settings_probab_mapping["features_config"] = "coord_features"
    default_bmfia_iterator.settings_probab_mapping['coords_cols'] = []
    with pytest.raises(AssertionError):
        z_mat = default_bmfia_iterator._set_feature_strategy(y_lf_mat, x_mat, coords_mat)

    # test coord_features without list format for cols
    default_bmfia_iterator.settings_probab_mapping["features_config"] = "coord_features"
    default_bmfia_iterator.settings_probab_mapping['coords_cols'] = 0
    with pytest.raises(AssertionError):
        z_mat = default_bmfia_iterator._set_feature_strategy(y_lf_mat, x_mat, coords_mat)

    # test coord_features with correct configuration
    expected_z_mat = np.array(
        [[[1, 2, 3], [1, 2, 3], [1, 2, 3]], [[7, 7, 7], [10, 10, 10], [13, 13, 13]]]
    )
    default_bmfia_iterator.settings_probab_mapping["features_config"] = "coord_features"
    default_bmfia_iterator.settings_probab_mapping['coords_cols'] = [0]
    z_mat = default_bmfia_iterator._set_feature_strategy(y_lf_mat, x_mat, coords_mat)
    np.testing.assert_array_almost_equal(z_mat, expected_z_mat, decimal=4)


def test_get_no_features(default_bmfia_iterator):
    """Test output without additional features."""
    y_lf_mat = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    x_mat = np.array([[4, 5, 6], [4, 5, 6], [4, 5, 6]])
    coords_mat = np.array([[7, 8, 9], [10, 11, 12], [13, 14, 15]])

    expected_z_mat = y_lf_mat
    default_bmfia_iterator.settings_probab_mapping["features_config"] = "no_features"
    z_mat = default_bmfia_iterator._set_feature_strategy(y_lf_mat, x_mat, coords_mat)
    np.testing.assert_array_almost_equal(z_mat, expected_z_mat, decimal=4)


def test_get_time_features(default_bmfia_iterator):
    """Test generation of time-based features."""
    y_lf_mat = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    x_mat = np.array([[4, 5, 6], [4, 5, 6], [4, 5, 6]])
    coords_mat = np.array([[7, 8, 9], [10, 11, 12], [13, 14, 15]])

    expected_z_mat = np.array([[1, 2, 3, 0], [1, 2, 3, 5], [1, 2, 3, 10]])
    default_bmfia_iterator.settings_probab_mapping["features_config"] = "time_features"
    default_bmfia_iterator.time_vec = np.linspace(0, 10, y_lf_mat.shape[1])
    z_mat = default_bmfia_iterator._set_feature_strategy(y_lf_mat, x_mat, coords_mat)
    np.testing.assert_array_almost_equal(z_mat, expected_z_mat, decimal=4)


def test_update_probabilistic_mapping_with_features(default_bmfia_iterator):
    """Test for updating with optimal informative features."""
    with pytest.raises(NotImplementedError):
        default_bmfia_iterator._update_probabilistic_mapping_with_features()


def test_eval_model(default_bmfia_iterator, mocker):
    """Test for evaluating the underlying model."""
    mo_1 = mocker.patch(
        'pqueens.iterators.bmfia_iterator.BMFIAIterator._evaluate_LF_model_for_X_train'
    )
    mo_2 = mocker.patch(
        'pqueens.iterators.bmfia_iterator.BMFIAIterator._evaluate_HF_model_for_X_train'
    )
    default_bmfia_iterator.eval_model()

    # --- asserts / tests ---
    mo_1.assert_called_once()
    mo_2.assert_called_once()
