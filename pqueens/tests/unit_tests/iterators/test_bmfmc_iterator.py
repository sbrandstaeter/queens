"""TODO_doc."""

import numpy as np
import pytest
from mock import Mock

from pqueens.distributions.uniform import UniformDistribution
from pqueens.interfaces.bmfmc_interface import BmfmcInterface
from pqueens.iterators.bmfmc_iterator import BMFMCIterator
from pqueens.parameters.parameters import Parameters


# ------ general input fixture ---------------------------------------
@pytest.fixture()
def default_parameters():
    """TODO_doc."""
    x1 = UniformDistribution(lower_bound=-2, upper_bound=2)
    x2 = UniformDistribution(lower_bound=-2, upper_bound=2)
    return Parameters(x1=x1, x2=x2)


@pytest.fixture()
def approx_name():
    """TODO_doc."""
    name = 'gp_approximation_gpflow'
    return name


@pytest.fixture()
def default_interface():
    """TODO_doc."""
    approx = "dummy_approx"
    interface = BmfmcInterface(approx)
    return interface


@pytest.fixture()
def config():
    """TODO_doc."""
    config = {
        "type": "gp_approximation_gpflow",
        "features_config": "opt_features",
        "num_features": 1,
        "X_cols": 1,
    }
    return config


@pytest.fixture()
def default_bmfmc_model(default_interface, default_parameters):
    """TODO_doc."""
    np.random.seed(1)
    model = Mock()
    model.X_mc = np.random.random((20, 2))
    model.Y_LFs_mc = np.random.random((20, 2))
    return model


@pytest.fixture()
def global_settings():
    """TODO_doc."""
    global_set = {'output_dir': 'dummyoutput', 'experiment_name': 'dummy_exp_name'}
    return global_set


@pytest.fixture()
def result_description():
    """TODO_doc."""
    description = {"write_results": True}
    return description


@pytest.fixture()
def experiment_dir():
    """TODO_doc."""
    return 'my_dummy_dir'


@pytest.fixture()
def initial_design():
    """TODO_doc."""
    return {"num_HF_eval": 5, "num_bins": 5, "method": "diverse_subset"}


@pytest.fixture()
def predictive_var():
    """TODO_doc."""
    return False


@pytest.fixture()
def BMFMC_reference():
    """TODO_doc."""
    return "dummy_reference"


@pytest.fixture()
def default_bmfmc_iterator(
    global_settings,
    default_parameters,
    default_bmfmc_model,
    result_description,
    initial_design,
    predictive_var,
    BMFMC_reference,
):
    """TODO_doc."""
    my_bmfmc_iterator = BMFMCIterator(
        model=default_bmfmc_model,
        global_settings=global_settings,
        parameters=default_parameters,
        result_description=result_description,
        initial_design=initial_design,
    )
    return my_bmfmc_iterator


# ------ actual unit_tests --------------------------------------------
def test_init(
    mocker,
    global_settings,
    default_parameters,
    default_bmfmc_model,
    result_description,
    initial_design,
):
    """TODO_doc."""
    mp = mocker.patch('pqueens.iterators.iterator.Iterator.__init__')
    my_bmfmc_iterator = BMFMCIterator(
        model=default_bmfmc_model,
        global_settings=global_settings,
        parameters=default_parameters,
        result_description=result_description,
        initial_design=initial_design,
    )
    # tests / asserts
    mp.assert_called_once_with(default_bmfmc_model, global_settings, default_parameters)
    assert my_bmfmc_iterator.result_description == result_description
    assert my_bmfmc_iterator.X_train is None
    assert my_bmfmc_iterator.Y_LFs_train is None
    assert my_bmfmc_iterator.output is None
    assert my_bmfmc_iterator.initial_design == initial_design


def test_core_run(mocker, default_bmfmc_iterator, default_bmfmc_model):
    """TODO_doc."""
    mp1 = mocker.patch('pqueens.iterators.bmfmc_iterator.BMFMCIterator.calculate_optimal_X_train')

    default_bmfmc_iterator.core_run()

    # -------- Load MC data from model -----------------------
    default_bmfmc_model.load_sampling_data.assert_called_once()

    # ---- determine optimal input points for which HF should be simulated -------
    mp1.assert_called_once()

    # ----- build model on training points and evaluate it -----------------------
    default_bmfmc_model.evaluate.assert_called_once()


def test_calculate_optimal_X_train(mocker, default_bmfmc_iterator):
    """TODO_doc."""
    mp1 = mocker.patch('pqueens.iterators.bmfmc_iterator.BMFMCIterator._diverse_subset_design')
    mocker.patch('pqueens.visualization.bmfmc_visualization.bmfmc_visualization_instance')

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
    """TODO_doc."""
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
    """TODO_doc."""
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
    """TODO_doc."""
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


def test_model_evaluate(default_bmfmc_iterator):
    """TODO_doc."""
    default_bmfmc_iterator.model.evaluate(None)
    default_bmfmc_iterator.model.evaluate.assert_called_once()


def test_post_run(mocker, default_bmfmc_iterator):
    """TODO_doc."""
    mocker.patch('pqueens.visualization.bmfmc_visualization.bmfmc_visualization_instance')

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
