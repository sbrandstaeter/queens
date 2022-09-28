import numpy as np
import pytest

import pqueens.parameters.parameters as parameters_module
from pqueens.interfaces.bmfmc_interface import BmfmcInterface
from pqueens.iterators.bmfmc_iterator import BMFMCIterator
from pqueens.models.bmfmc_model import BMFMCModel


# ------ general input fixture ---------------------------------------
@pytest.fixture()
def default_parameters():
    params = {
        "random_variables": {
            "x1": {"type": "FLOAT", "dimension": 1, "lower_bound": -2.0, "upper_bound": 2.0},
            "x2": {"type": "FLOAT", "dimension": 1, "lower_bound": -2.0, "upper_bound": 2.0},
        }
    }
    parameters_module.from_config_create_parameters({"parameters": params})
    return params


@pytest.fixture()
def approx_name():
    name = 'gp_approximation_gpy'
    return name


@pytest.fixture()
def default_interface(config):
    interface = BmfmcInterface(config, approx_name)
    return interface


@pytest.fixture()
def config():
    config = {
        "type": "gp_approximation_gpy",
        "features_config": "opt_features",
        "num_features": 1,
        "X_cols": 1,
    }
    return config


@pytest.fixture()
def default_bmfmc_model(default_interface, default_parameters):
    settings_probab_mapping = {
        "type": "gp_approximation_gpy",
        "features_config": "opt_features",
        "num_features": 1,
    }
    y_pdf_support = np.linspace(-1, 1, 10)
    uncertain_parameters = None

    model = BMFMCModel(
        settings_probab_mapping,
        False,
        y_pdf_support,
        default_interface,
        hf_model=None,
        no_features_comparison_bool=False,
        lf_data_iterators=None,
        hf_data_iterator=None,
    )
    np.random.seed(1)
    model.X_mc = np.random.random((20, 2))
    model.Y_LFs_mc = np.random.random((20, 2))

    return model


@pytest.fixture()
def global_settings():
    global_set = {'output_dir': 'dummyoutput', 'experiment_name': 'dummy_exp_name'}
    return global_set


@pytest.fixture()
def result_description():
    description = {"write_results": True}
    return description


@pytest.fixture()
def experiment_dir():
    return 'my_dummy_dir'


@pytest.fixture()
def initial_design():
    return {"num_HF_eval": 5, "num_bins": 5, "method": "diverse_subset"}


@pytest.fixture()
def predictive_var():
    return False


@pytest.fixture()
def BMFMC_reference():
    return "dummy_reference"


@pytest.fixture()
def default_bmfmc_iterator(
    global_settings,
    default_bmfmc_model,
    result_description,
    initial_design,
    predictive_var,
    BMFMC_reference,
):
    my_bmfmc_iterator = BMFMCIterator(
        default_bmfmc_model,
        result_description,
        initial_design,
        predictive_var,
        BMFMC_reference,
        global_settings,
    )
    return my_bmfmc_iterator


# custom class to mock the visualization module
class InstanceMock:
    @staticmethod
    def plot_pdfs(self, *args, **kwargs):
        return 1

    @staticmethod
    def plot_manifold(self, *args, **kwargs):
        return 1

    @staticmethod
    def plot_feature_ranking(self, *args, **kwargs):
        return 1


@pytest.fixture
def mock_visualization():
    my_mock = InstanceMock()
    return my_mock


# ------ actual unit_tests --------------------------------------------
def test_init(
    mocker,
    global_settings,
    default_bmfmc_model,
    result_description,
    initial_design,
    predictive_var,
    BMFMC_reference,
):
    mp = mocker.patch('pqueens.iterators.iterator.Iterator.__init__')
    my_bmfmc_iterator = BMFMCIterator(
        default_bmfmc_model,
        result_description,
        initial_design,
        predictive_var,
        BMFMC_reference,
        global_settings,
    )
    # tests / asserts
    mp.assert_called_once_with(None, global_settings)
    assert my_bmfmc_iterator.model == default_bmfmc_model
    assert my_bmfmc_iterator.result_description == result_description
    assert my_bmfmc_iterator.X_train is None
    assert my_bmfmc_iterator.Y_LFs_train is None
    assert my_bmfmc_iterator.output is None
    assert my_bmfmc_iterator.initial_design == initial_design
    assert my_bmfmc_iterator.predictive_var == predictive_var
    assert my_bmfmc_iterator.BMFMC_reference == BMFMC_reference


def test_core_run(mocker, default_bmfmc_iterator, default_bmfmc_model):
    mp1 = mocker.patch('pqueens.models.bmfmc_model.BMFMCModel.load_sampling_data')
    mp2 = mocker.patch('pqueens.iterators.bmfmc_iterator.BMFMCIterator.calculate_optimal_X_train')
    mp3 = mocker.patch('pqueens.models.bmfmc_model.BMFMCModel.evaluate')

    default_bmfmc_iterator.core_run()

    # -------- Load MC data from model -----------------------
    mp1.assert_called_once()

    # ---- determine optimal input points for which HF should be simulated -------
    mp2.assert_called_once()

    # ----- build model on training points and evaluate it -----------------------
    mp3.assert_called_once()


def test_calculate_optimal_X_train(mocker, default_bmfmc_iterator):
    mp1 = mocker.patch('pqueens.iterators.bmfmc_iterator.BMFMCIterator._diverse_subset_design')
    mocker.patch(
        'pqueens.visualization.bmfmc_visualization.bmfmc_visualization_instance',
        return_value=mock_visualization,
    )

    mocker.patch(
        'pqueens.visualization.bmfmc_visualization.bmfmc_visualization_instance'
        '.plot_feature_ranking'
    )
    default_bmfmc_iterator.X_train = np.array([1.0, 1.0, 1.0])
    default_bmfmc_iterator.Y_LFs_train = np.array([2.0, 2.0, 2.0])
    default_bmfmc_iterator.calculate_optimal_X_train()

    mp1.assert_called_once()
    np.testing.assert_array_equal(default_bmfmc_iterator.model.X_train, np.array([1.0, 1.0, 1.0]))
    np.testing.assert_array_equal(
        default_bmfmc_iterator.model.Y_LFs_train, np.array([2.0, 2.0, 2.0])
    )


def test_get_design_method(mocker, default_bmfmc_iterator):
    mocker.patch(
        'pqueens.iterators.bmfmc_iterator.BMFMCIterator._random_design', return_value='random'
    )
    mocker.patch(
        'pqueens.iterators.bmfmc_iterator.BMFMCIterator._diverse_subset_design',
        return_value='diverse',
    )
    mocker.patch('pqueens.models.bmfmc_model.BMFMCModel.calculate_extended_gammas')

    method = default_bmfmc_iterator._get_design_method('random')
    assert method() == 'random'

    method = default_bmfmc_iterator._get_design_method('diverse_subset')
    assert method() == 'diverse'

    with pytest.raises(NotImplementedError):
        default_bmfmc_iterator._get_design_method('blabla')


def test_diverse_subset_design(mocker, default_bmfmc_iterator):
    mocker.patch('pqueens.models.bmfmc_model.BMFMCModel.calculate_extended_gammas')
    n_points = 3
    np.random.seed(1)
    default_bmfmc_iterator.model.gammas_ext_mc = np.random.random((10, 2))
    default_bmfmc_iterator._diverse_subset_design(n_points)
    X_train = default_bmfmc_iterator.X_train
    Y_LFs_train = default_bmfmc_iterator.Y_LFs_train

    expected_X_train = np.array(
        np.array([[0.20445225, 0.87811744], [0.4173048, 0.55868983], [0.14038694, 0.19810149]])
    )
    expected_Y_LFs_train = np.array(
        np.array([[0.21162812, 0.26554666], [0.57411761, 0.14672857], [0.58930554, 0.69975836]])
    )
    np.testing.assert_array_almost_equal(X_train, expected_X_train, decimal=6)
    np.testing.assert_array_almost_equal(Y_LFs_train, expected_Y_LFs_train, decimal=6)


def test_random_design(mocker, default_bmfmc_iterator):
    mocker.patch('pqueens.models.bmfmc_model.BMFMCModel.calculate_extended_gammas')
    n_points = 3
    np.random.seed(1)
    default_bmfmc_iterator.model.gammas_ext_mc = np.random.random((10, 2))
    default_bmfmc_iterator.initial_design = {
        "num_HF_eval": 3,
        "num_bins": 3,
        "method": "random",
        "seed": 1,
        "master_LF": 1,
    }
    default_bmfmc_iterator._random_design(n_points)
    X_train = default_bmfmc_iterator.X_train
    Y_LFs_train = default_bmfmc_iterator.Y_LFs_train

    expected_X_train = np.array(
        np.array([[0.6918771, 0.3155156], [0.3134242, 0.6923226], [0.1403869, 0.1981015]])
    )
    expected_Y_LFs_train = np.array(
        np.array([[0.3976768, 0.1653542], [0.6944002, 0.4141793], [0.5893055, 0.6997584]])
    )
    np.testing.assert_array_almost_equal(X_train, expected_X_train, decimal=6)
    np.testing.assert_array_almost_equal(Y_LFs_train, expected_Y_LFs_train, decimal=6)


def test_model_evaluate(mocker, default_bmfmc_iterator):
    mp1 = mocker.patch('pqueens.models.bmfmc_model.BMFMCModel.evaluate')
    default_bmfmc_iterator.model.evaluate(None)
    mp1.assert_called_once()


def test_post_run(mocker, default_bmfmc_iterator):
    mocker.patch(
        'pqueens.visualization.bmfmc_visualization.bmfmc_visualization_instance',
        return_value=mock_visualization,
    )

    mp1 = mocker.patch(
        'pqueens.visualization.bmfmc_visualization.bmfmc_visualization_instance.plot_pdfs'
    )

    mp2 = mocker.patch(
        'pqueens.visualization.bmfmc_visualization.bmfmc_visualization_instance.plot_manifold'
    )
    mp3 = mocker.patch('pqueens.iterators.bmfmc_iterator.write_results')

    default_bmfmc_iterator.post_run()

    mp1.assert_called_once()
    mp2.assert_called_once()
    mp3.assert_called_once()
