"""Integration tests for the BMFIA."""

# pylint: disable=invalid-name
import pickle

import numpy as np
import pytest

from queens.distributions.normal import NormalDistribution
from queens.distributions.uniform import UniformDistribution
from queens.interfaces.bmfia_interface import BmfiaInterface
from queens.interfaces.direct_python_interface import DirectPythonInterface
from queens.iterators.bmfia_iterator import BMFIAIterator
from queens.iterators.reparameteriztion_based_variational_inference import RPVIIterator
from queens.iterators.sequential_monte_carlo_iterator import SequentialMonteCarloIterator
from queens.main import run_iterator
from queens.models.likelihood_models.bayesian_mf_gaussian_likelihood import BMFGaussianModel
from queens.models.simulation_model import SimulationModel
from queens.models.surrogate_models.gaussian_neural_network import GaussianNeuralNetworkModel
from queens.models.surrogate_models.gp_approximation_jitted import GPJittedModel
from queens.parameters.parameters import Parameters
from queens.utils.experimental_data_reader import ExperimentalDataReader
from queens.utils.stochastic_optimizer import Adam


@pytest.mark.max_time_for_test(30)
def test_bmfia_smc_park(
    tmp_path,
    _create_experimental_data_park91a_hifi_on_grid,
    expected_samples,
    expected_weights,
    _initialize_global_settings,
):
    """Integration test for BMFIA.

    Integration test for bayesian multi-fidelity inverse analysis
    (bmfia) using the park91 function.
    """
    # generate yml input file from template

    experimental_data_path = tmp_path
    plot_dir = tmp_path

    # Parameters
    x1 = UniformDistribution(lower_bound=0.01, upper_bound=0.99)
    x2 = UniformDistribution(lower_bound=0.01, upper_bound=0.99)
    parameters = Parameters(x1=x1, x2=x2)

    # Setup QUEENS stuff
    experimental_data_reader = ExperimentalDataReader(
        file_name_identifier="*.csv",
        csv_data_base_dir=experimental_data_path,
        output_label="y_obs",
        coordinate_labels=["x3", "x4"],
    )
    mf_interface = BmfiaInterface(
        num_processors_multi_processing=4,
        probabilistic_mapping_type="per_coordinate",
        parameters=parameters,
    )
    stochastic_optimizer = Adam(
        learning_rate=0.008,
        optimization_type="max",
        max_iteration=1000,
        rel_l1_change_threshold=0.004,
        rel_l2_change_threshold=0.004,
    )
    mf_approx = GPJittedModel(
        kernel_type="squared_exponential",
        plot_refresh_rate=None,
        noise_var_lb=1e-06,
        data_scaling="standard_scaler",
        initial_hyper_params_lst=[0.05, 1.0, 0.05],
        mean_function_type="identity_multi_fidelity",
        stochastic_optimizer=stochastic_optimizer,
    )
    mcmc_proposal_distribution = NormalDistribution(
        mean=[0.0, 0.0], covariance=[[0.01, 0.0], [0.0, 0.01]]
    )
    interface = DirectPythonInterface(
        num_workers=1, function="park91a_lofi_on_grid", parameters=parameters
    )
    forward_model = SimulationModel(interface=interface)
    lf_model = SimulationModel(interface=interface)
    interface = DirectPythonInterface(
        num_workers=1, function="park91a_hifi_on_grid", parameters=parameters
    )
    hf_model = SimulationModel(interface=interface)
    mf_subiterator = BMFIAIterator(
        features_config="man_features",
        X_cols=[0],
        num_features=1,
        initial_design={"type": "random", "num_HF_eval": 20, "seed": 1},
        lf_model=lf_model,
        hf_model=hf_model,
        parameters=parameters,
    )
    model = BMFGaussianModel(
        noise_value=0.001,
        plotting_options={
            "plotting_dir": plot_dir,
            "plot_names": ["bmfia_plot.png", "posterior.png"],
            "save_bool": [False, False],
            "plot_booleans": [False, False],
        },
        experimental_data_reader=experimental_data_reader,
        mf_interface=mf_interface,
        mf_approx=mf_approx,
        forward_model=forward_model,
        mf_subiterator=mf_subiterator,
    )
    iterator = SequentialMonteCarloIterator(
        seed=41,
        num_particles=10,
        temper_type="bayes",
        plot_trace_every=0,
        num_rejuvenation_steps=2,
        result_description={"write_results": True, "plot_results": False, "cov": True},
        mcmc_proposal_distribution=mcmc_proposal_distribution,
        model=model,
        parameters=parameters,
    )

    # Actual analysis
    run_iterator(iterator)

    # Load results
    result_file = tmp_path / "dummy_experiment_name.pickle"

    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    samples = results['raw_output_data']['particles'].squeeze()
    weights = results['raw_output_data']['weights'].squeeze()

    # some tests / asserts here
    np.testing.assert_array_almost_equal(samples, expected_samples, decimal=5)
    np.testing.assert_array_almost_equal(weights.flatten(), expected_weights.flatten(), decimal=5)


@pytest.fixture(name="expected_samples")
def fixture_expected_samples():
    """Fixture for expected SMC samples."""
    samples = np.array(
        [
            [0.51711296, 0.55200585],
            [0.4996905, 0.6673229],
            [0.48662203, 0.68802404],
            [0.49806929, 0.66276797],
            [0.49706481, 0.68586978],
            [0.50424704, 0.65139028],
            [0.51437955, 0.57678317],
            [0.51275639, 0.58981357],
            [0.50163956, 0.65389397],
            [0.52127371, 0.61237995],
        ]
    )

    return samples


@pytest.fixture(name="expected_weights")
def fixture_expected_weights():
    """Fixture for expected SMC weights."""
    weights = np.array(
        [
            0.00183521,
            0.11284748,
            0.16210619,
            0.07066473,
            0.10163831,
            0.09845534,
            0.10742886,
            0.15461861,
            0.09222745,
            0.0981778,
        ]
    )
    return weights


@pytest.mark.max_time_for_test(20)
def test_bmfia_rpvi_gp_park(
    tmp_path,
    _create_experimental_data_park91a_hifi_on_grid,
    expected_variational_mean,
    expected_variational_cov,
    _initialize_global_settings,
):
    """Integration test for BMFIA.

    Integration test for bayesian multi-fidelity inverse analysis
    (bmfia) using the park91 function.
    """
    experimental_data_path = tmp_path

    # Parameters
    x1 = UniformDistribution(lower_bound=0.01, upper_bound=0.99)
    x2 = UniformDistribution(lower_bound=0.01, upper_bound=0.99)
    parameters = Parameters(x1=x1, x2=x2)

    # Setup QUEENS stuff
    stochastic_optimizer = Adam(
        learning_rate=0.03,
        max_iteration=10000000,
        optimization_type="max",
        rel_l1_change_threshold=-1,
        rel_l2_change_threshold=-1,
    )
    experimental_data_reader = ExperimentalDataReader(
        file_name_identifier="*.csv",
        csv_data_base_dir=experimental_data_path,
        output_label="y_obs",
        coordinate_labels=["x3", "x4"],
    )
    mf_interface = BmfiaInterface(
        num_processors_multi_processing=4,
        probabilistic_mapping_type="per_coordinate",
        parameters=parameters,
    )
    stochastic_optimizer = Adam(
        learning_rate=0.008,
        optimization_type="max",
        rel_l1_change_threshold=0.004,
        rel_l2_change_threshold=0.004,
    )
    mf_approx = GPJittedModel(
        data_scaling="standard_scaler",
        initial_hyper_params_lst=[0.05, 1.0, 0.05],
        kernel_type="squared_exponential",
        noise_var_lb=1e-06,
        mean_function_type="identity_multi_fidelity",
        stochastic_optimizer=stochastic_optimizer,
    )
    interface = DirectPythonInterface(
        function="park91a_hifi_on_grid", num_workers=1, parameters=parameters
    )
    hf_model = SimulationModel(interface=interface)
    interface = DirectPythonInterface(
        function="park91a_lofi_on_grid_with_gradients",
        num_workers=1,
        parameters=parameters,
    )
    forward_model = SimulationModel(interface=interface)
    lf_model = SimulationModel(interface=interface)
    mf_subiterator = BMFIAIterator(
        features_config="man_features",
        num_features=1,
        X_cols=[0],
        initial_design={"num_HF_eval": 50, "seed": 1, "type": "random"},
        hf_model=hf_model,
        lf_model=lf_model,
        parameters=parameters,
    )
    model = BMFGaussianModel(
        noise_value=0.0001,
        plotting_options={
            "plot_booleans": [False, False],
            "plot_names": ["bmfia_plot.png", "posterior.png"],
            "plotting_dir": "plot_dir",
            "save_bool": [False, False],
        },
        experimental_data_reader=experimental_data_reader,
        mf_interface=mf_interface,
        mf_approx=mf_approx,
        forward_model=forward_model,
        mf_subiterator=mf_subiterator,
    )
    method = RPVIIterator(
        max_feval=100,
        n_samples_per_iter=3,
        random_seed=1,
        result_description={
            "iterative_field_names": ["variational_parameters", "elbo"],
            "plotting_options": {
                "plot_boolean": False,
                "plot_name": "variational_params_convergence.jpg",
                "plot_refresh_rate": None,
                "plotting_dir": "plot_dir",
                "save_bool": False,
            },
            "write_results": True,
        },
        score_function_bool=False,
        variational_distribution={
            "variational_approximation_type": "mean_field",
            "variational_family": "normal",
        },
        variational_parameter_initialization="prior",
        variational_transformation=None,
        stochastic_optimizer=stochastic_optimizer,
        model=model,
        parameters=parameters,
    )

    # Actual analysis
    run_iterator(method)

    # Load results
    result_file = tmp_path / "dummy_experiment_name.pickle"

    # get the results of the QUEENS run
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    variational_mean = results['variational_distribution']['mean']
    variational_cov = results['variational_distribution']['covariance']

    # some tests / asserts here
    np.testing.assert_array_almost_equal(variational_mean, expected_variational_mean, decimal=2)
    np.testing.assert_array_almost_equal(variational_cov, expected_variational_cov, decimal=2)


@pytest.fixture(name="expected_variational_mean")
def fixture_expected_variational_mean():
    """Fixture for expected variational_mean."""
    exp_var_mean = np.array([0.51, 0.5]).reshape(-1, 1)

    return exp_var_mean


@pytest.fixture(name="expected_variational_cov")
def fixture_expected_variational_cov():
    """Fixture for expected variational covariance."""
    exp_var_cov = np.array([[0.00142648, 0.0], [0.0, 0.00347234]])
    return exp_var_cov


def test_bmfia_rpvi_nn_park(
    tmp_path,
    _create_experimental_data_park91a_hifi_on_grid,
    expected_variational_mean_nn,
    expected_variational_cov_nn,
    _initialize_global_settings,
):
    """Integration test for BMFIA.

    Integration test for bayesian multi-fidelity inverse analysis
    (bmfia) using the park91 function.
    """
    experimental_data_path = tmp_path
    plot_dir = tmp_path

    # Parameters
    x1 = NormalDistribution(covariance=0.09, mean=0.5)
    x2 = NormalDistribution(covariance=0.09, mean=0.5)
    parameters = Parameters(x1=x1, x2=x2)

    # Setup QUEENS stuff
    experimental_data_reader = ExperimentalDataReader(
        file_name_identifier="*.csv",
        csv_data_base_dir=experimental_data_path,
        output_label="y_obs",
        coordinate_labels=["x3", "x4"],
    )
    mf_approx = GaussianNeuralNetworkModel(
        activation_per_hidden_layer_lst=["elu", "elu"],
        adams_training_rate=0.001,
        data_scaling="standard_scaler",
        nodes_per_hidden_layer_lst=[5, 5],
        nugget_std=1e-05,
        num_epochs=1,
        optimizer_seed=42,
        refinement_epochs_decay=0.7,
        verbosity_on=True,
    )
    mf_interface = BmfiaInterface(
        num_processors_multi_processing=1,
        probabilistic_mapping_type="per_time_step",
        parameters=parameters,
    )
    interface = DirectPythonInterface(
        function="park91a_hifi_on_grid", num_workers=1, parameters=parameters
    )
    hf_model = SimulationModel(interface=interface)
    interface = DirectPythonInterface(
        function="park91a_lofi_on_grid_with_gradients",
        num_workers=1,
        parameters=parameters,
    )
    forward_model = SimulationModel(interface=interface)
    lf_model = SimulationModel(interface=interface)
    mf_subiterator = BMFIAIterator(
        features_config="no_features",
        initial_design={"num_HF_eval": 50, "seed": 1, "type": "random"},
        hf_model=hf_model,
        lf_model=lf_model,
        parameters=parameters,
    )
    model = BMFGaussianModel(
        noise_value=0.0001,
        plotting_options={
            "plot_booleans": [False, False],
            "plot_names": ["bmfia_plot.png", "posterior.png"],
            "plot_options_dict": {"x_lim": [0, 10], "y_lim": [0, 10]},
            "plotting_dir": plot_dir,
            "save_bool": [False, False],
        },
        experimental_data_reader=experimental_data_reader,
        mf_approx=mf_approx,
        mf_interface=mf_interface,
        forward_model=forward_model,
        mf_subiterator=mf_subiterator,
    )
    stochastic_optimizer = Adam(
        learning_rate=0.01,
        max_iteration=100,
        optimization_type="max",
        rel_l1_change_threshold=-1,
        rel_l2_change_threshold=-1,
    )
    method = RPVIIterator(
        max_feval=100,
        n_samples_per_iter=3,
        random_seed=1,
        result_description={
            "iterative_field_names": ["variational_parameters", "elbo"],
            "plotting_options": {
                "plot_boolean": False,
                "plot_name": "variational_params_convergence.jpg",
                "plot_refresh_rate": None,
                "plotting_dir": plot_dir,
                "save_bool": False,
            },
            "write_results": True,
        },
        score_function_bool=False,
        variational_distribution={
            "variational_approximation_type": "mean_field",
            "variational_family": "normal",
        },
        variational_parameter_initialization="prior",
        variational_transformation=None,
        model=model,
        stochastic_optimizer=stochastic_optimizer,
        parameters=parameters,
    )

    # Actual analysis
    run_iterator(method)

    # Load results
    result_file = tmp_path / "dummy_experiment_name.pickle"
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    variational_mean = results['variational_distribution']['mean']
    variational_cov = results['variational_distribution']['covariance']

    # some tests / asserts here
    np.testing.assert_array_almost_equal(variational_mean, expected_variational_mean_nn, decimal=1)
    np.testing.assert_array_almost_equal(variational_cov, expected_variational_cov_nn, decimal=1)


@pytest.fixture(name="expected_variational_mean_nn")
def fixture_expected_variational_mean_nn():
    """Fixture for expected variational_mean."""
    exp_var_mean = np.array([0.19221321, 0.33134219]).reshape(-1, 1)

    return exp_var_mean


@pytest.fixture(name="expected_variational_cov_nn")
def fixture_expected_variational_cov_nn():
    """Fixture for expected variational covariance."""
    exp_var_cov = np.array([[0.01245263, 0.0], [0.0, 0.01393423]])
    return exp_var_cov
