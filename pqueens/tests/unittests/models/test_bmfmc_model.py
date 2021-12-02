import pytest
import numpy as np
from mock import patch
from pqueens.models.bmfmc_model import BMFMCModel
from pqueens.models.simulation_model import SimulationModel
import pqueens.models.bmfmc_model as bmfmc_model
from pqueens.interfaces.bmfmc_interface import BmfmcInterface
from pqueens.iterators.data_iterator import DataIterator


# ------------ fixtures --------------------------
@pytest.fixture()
def result_description():
    description = {"write_results": True}
    return description


@pytest.fixture()
def dummy_high_fidelity_model(parameters):
    model_name = 'dummy'
    interface = 'my_dummy_interface'
    model_parameters = parameters
    hf_model = SimulationModel(model_name, interface, model_parameters)
    hf_model.response = {'mean': 1.0}
    return hf_model


@pytest.fixture()
def global_settings():
    global_set = {'output_dir': 'dummyoutput', 'experiment_name': 'dummy_exp_name'}
    return global_set


@pytest.fixture()
def parameters():
    params = {
        "random_variables": {
            "x1": {"type": "FLOAT", "size": 1, "min": -2.0, "max": 2.0},
            "x2": {"type": "FLOAT", "size": 1, "min": -2.0, "max": 2.0},
        },
        "random_fields": {
            'random_inflow': {
                'type': 'FLOAT',
                'dimension': 1,
                'min': 0,
                'max': 1,
                'corrstruct': 'non_stationary_squared_exp',
                'corr_length': 0.08,
                'std_hyperparam_rf': 0.1,
                'mean_fun': 'inflow_parabola',
                'mean_fun_params': [1.5],
                'num_points': 10,
            }
        },
    }
    return params


@pytest.fixture()
def config():
    config = {
        "joint_density_approx": {
            "type": "gp_approximation_gpy",
            "features_config": "opt_features",
            "num_features": 1,
            "X_cols": 1,
        }
    }
    return config


@pytest.fixture()
def approximation_name():
    name = 'joint_density_approx'
    return name


@pytest.fixture()
def default_interface(config, approximation_name):
    interface = BmfmcInterface(config, approximation_name)
    return interface


@pytest.fixture()
def settings_probab_mapping(config, approximation_name):
    settings = config[approximation_name]
    return settings


@pytest.fixture()
def default_bmfmc_model(parameters, settings_probab_mapping, default_interface):
    y_pdf_support = np.linspace(-1, 1, 10)

    model = BMFMCModel(
        settings_probab_mapping,
        False,
        y_pdf_support,
        parameters,
        default_interface,
        hf_model=None,
        no_features_comparison_bool=True,
        lf_data_iterators=None,
        hf_data_iterator=None,
    )

    np.random.seed(1)
    model.X_mc = np.random.random((20, 2))
    np.random.seed(2)
    model.Y_LFs_mc = np.random.random((20, 2))
    np.random.seed(3)
    model.Z_mc = np.random.random((20, 2))
    model.Y_HF_mc = np.array([1, 1])

    return model


@pytest.fixture()
def default_data_iterator(result_description, global_settings):
    path_to_data = 'dummy'
    data_iterator = DataIterator(path_to_data, result_description, global_settings)
    return data_iterator


@pytest.fixture()
def dummy_MC_data(parameters):
    data_dict = {
        "uncertain_parameters": parameters,
        "input": np.array([[1.0, 1.0]]),
        "eigenfunc": np.array([[1.0, 1.0]]),
        "eigenvalue": np.array([[1.0, 1.0]]),
        "output": np.array([[1.0, 1.0], [2.0, 2.0]]).T,
    }

    return data_dict


# custom class to mock the visualization module
class InstanceMock:
    @staticmethod
    def plot_feature_ranking(self, *args, **kwargs):
        return 1


@pytest.fixture
def mock_visualization():
    my_mock = InstanceMock()
    return my_mock


# ------------ unittests -------------------------
@pytest.mark.unit_tests
def test_init(mocker, settings_probab_mapping, default_interface):
    y_pdf_support = np.linspace(-1, 1, 10)

    mp = mocker.patch('pqueens.models.model.Model.__init__')

    model = BMFMCModel(
        settings_probab_mapping,
        True,
        y_pdf_support,
        parameters,
        default_interface,
        hf_model=None,
        no_features_comparison_bool=False,
        lf_data_iterators=None,
        hf_data_iterator=None,
    )

    # tests/ asserts ------------------
    mp.assert_called_once_with(name='bmfmc_model', uncertain_parameters=parameters, data_flag=True)
    assert model.interface == default_interface
    assert model.settings_probab_mapping == settings_probab_mapping
    assert model.high_fidelity_model is None
    assert model.X_train is None
    assert model.Y_HF_train is None
    assert model.Y_LFs_train is None
    assert model.X_mc is None
    assert model.Y_LFs_mc is None
    assert model.Y_HF_mc is None
    assert model.gammas_ext_mc is None
    assert model.gammas_ext_train is None
    assert model.Z_train is None
    assert model.Z_mc is None
    assert model.m_f_mc is None
    assert model.var_y_mc is None
    assert model.p_yhf_mean is None
    assert model.p_yhf_var is None
    assert model.predictive_var_bool is True
    assert model.p_yhf_mc is None
    assert model.p_ylf_mc is None
    assert model.no_features_comparison_bool is False
    assert model.eigenfunc_random_fields is None
    assert model.eigenvals is None
    assert model.f_mean_train is None
    np.testing.assert_array_almost_equal(model.y_pdf_support, y_pdf_support, decimal=6)
    assert model.lf_data_iterators is None
    assert model.hf_data_iterator is None
    assert model.training_indices is None


@pytest.mark.unit_tests
def test_evaluate(mocker, default_bmfmc_model):
    mp1 = mocker.patch('pqueens.models.bmfmc_model.BMFMCModel.compute_pymc_reference')
    mp2 = mocker.patch(
        'pqueens.models.bmfmc_model.BMFMCModel.run_BMFMC_without_features', return_value=(1, 1)
    )
    mp3 = mocker.patch('pqueens.models.bmfmc_model.BMFMCModel.run_BMFMC')

    # create expected output
    default_bmfmc_model.Z_mc = np.array([1])
    default_bmfmc_model.m_f_mc = np.array([1])
    default_bmfmc_model.var_y_mc = np.array([1])
    default_bmfmc_model.y_pdf_support = np.array([1])
    default_bmfmc_model.p_yhf_mean = np.array([1])
    default_bmfmc_model.p_yhf_var = np.array([1])
    default_bmfmc_model.p_yhf_mean_BMFMC = np.array([1])
    default_bmfmc_model.p_yhf_var_BMFMC = np.array([1])
    default_bmfmc_model.p_ylf_mc = np.array([1])
    default_bmfmc_model.p_yhf_mc = np.array([1])
    default_bmfmc_model.Z_train = np.array([1])
    default_bmfmc_model.X_train = np.array([1])
    default_bmfmc_model.Y_HF_train = np.array([1])

    expected_output = {
        "Z_mc": np.array([1]),
        "m_f_mc": np.array([1]),
        "var_y_mc": np.array([1]),
        "y_pdf_support": np.array([1]),
        "p_yhf_mean": np.array([1]),
        "p_yhf_var": np.array([1]),
        "p_yhf_mean_BMFMC": np.array([1]),
        "p_yhf_var_BMFMC": np.array([1]),
        "p_ylf_mc": np.array([1]),
        "p_yhf_mc": np.array([1]),
        "Z_train": np.array([1]),
        "X_train": np.array([1]),
        "Y_HF_train": np.array([1]),
    }
    output = default_bmfmc_model.evaluate()

    mp1.assert_called_once()
    mp2.assert_called_once()
    mp3.assert_called_once()
    assert output == expected_output


@pytest.mark.unit_tests
def test_run_BMFMC(mocker, default_bmfmc_model):
    mp1 = mocker.patch('pqueens.models.bmfmc_model.BMFMCModel.build_approximation')
    mp2 = mocker.patch('pqueens.models.bmfmc_model.BMFMCModel.compute_pyhf_statistics')
    mp3 = mocker.patch('pqueens.interfaces.bmfmc_interface.BmfmcInterface.map', return_value=(1, 1))

    default_bmfmc_model.Z_mc = np.array([[1, 1]])
    default_bmfmc_model.Z_train = np.array([[1, 1]])
    default_bmfmc_model.run_BMFMC()

    mp1.assert_called_once()
    mp2.assert_called_once()
    assert mp3.call_count == 2


@pytest.mark.unit_tests
def test_run_BMFMC_without_features(mocker, default_bmfmc_model):
    mp1 = mocker.patch('pqueens.models.bmfmc_model.BMFMCModel.build_approximation')
    mp2 = mocker.patch('pqueens.models.bmfmc_model.BMFMCModel.compute_pyhf_statistics')
    mp3 = mocker.patch('pqueens.interfaces.bmfmc_interface.BmfmcInterface.map', return_value=(1, 1))

    default_bmfmc_model.Y_LFs_mc = np.array([[1, 1]])
    default_bmfmc_model.Y_LFs_train = np.array([[1, 1]])
    default_bmfmc_model.run_BMFMC_without_features()

    mp1.assert_called_once()
    mp2.assert_called_once()
    assert mp3.call_count == 2


@pytest.mark.unit_tests
def test_load_sampling_data(mocker, default_bmfmc_model, default_data_iterator, dummy_MC_data):
    mp1 = mocker.patch(
        'pqueens.iterators.data_iterator.DataIterator.read_pickle_file', return_value=dummy_MC_data
    )
    # modify default model to expect HF MC reference data
    default_bmfmc_model.lf_data_iterators = [default_data_iterator, default_data_iterator]
    default_bmfmc_model.hf_data_iterator = default_data_iterator

    # run the current method
    default_bmfmc_model.load_sampling_data()

    # tests min amount of data iterator calls
    assert mp1.call_count >= 6

    # test assembling of multiple LF data
    np.testing.assert_array_almost_equal(
        default_bmfmc_model.Y_LFs_mc, np.array([[1.0, 1.0], [1.0, 1.0]]), decimal=5
    )

    # test hf data iterator
    np.testing.assert_array_almost_equal(
        default_bmfmc_model.Y_HF_mc, np.array([1.0, 1.0]).T, decimal=5
    )


@pytest.mark.unit_tests
def test_get_hf_training_data(mocker, default_bmfmc_model, dummy_high_fidelity_model):
    mp1 = mocker.patch(
        'pqueens.models.simulation_model.SimulationModel' '.update_model_from_sample_batch'
    )
    mp2 = mocker.patch('pqueens.models.simulation_model.SimulationModel.evaluate')
    default_bmfmc_model.X_train = None

    # test X_train is None
    with pytest.raises(ValueError):
        default_bmfmc_model.get_hf_training_data()

    default_bmfmc_model.X_train = np.array([[1.0, 1.0], [2.0, 2.0]])
    default_bmfmc_model.Y_HF_mc = None

    # test HF model given
    default_bmfmc_model.high_fidelity_model = dummy_high_fidelity_model
    default_bmfmc_model.get_hf_training_data()
    mp1.assert_called_once()
    mp2.assert_called_once()

    # test HF MC data available and no HF model
    default_bmfmc_model.high_fidelity_model = None
    default_bmfmc_model.Y_HF_mc = np.array([2.1, 1.5, 6.4, 2.4])
    default_bmfmc_model.X_mc = np.array([[1.1, 1.2], [1.3, 1.4], [1.5, 1.6], [1.7, 1.8]])
    default_bmfmc_model.X_train = np.array([[1.3, 1.4], [1.5, 1.6]])
    default_bmfmc_model.get_hf_training_data()
    np.testing.assert_array_almost_equal(
        default_bmfmc_model.Y_HF_train, np.array([[1.5, 6.4]]).T, decimal=5
    )

    # test HF MC data available and HF model too (else statement)
    default_bmfmc_model.high_fidelity_model = dummy_high_fidelity_model
    with pytest.raises(RuntimeError):
        default_bmfmc_model.get_hf_training_data()


@pytest.mark.unit_tests
def test_build_approximation(mocker, default_bmfmc_model):
    mp1 = mocker.patch('pqueens.models.bmfmc_model.BMFMCModel.get_hf_training_data')
    mp2 = mocker.patch('pqueens.models.bmfmc_model.BMFMCModel.set_feature_strategy')
    mp3 = mocker.patch('pqueens.interfaces.bmfmc_interface.BmfmcInterface.build_approximation')

    default_bmfmc_model.eval_fit = 'kfold'
    default_bmfmc_model.build_approximation(approx_case=True)

    mp1.assert_called_once()
    mp2.assert_called_once()
    mp3.assert_called_once()


@pytest.mark.unit_tests
def test_compute_pyhf_statistics(mocker, default_bmfmc_model):
    mp1 = mocker.patch('pqueens.models.bmfmc_model.BMFMCModel._calculate_p_yhf_mean')
    mp2 = mocker.patch('pqueens.models.bmfmc_model.BMFMCModel._calculate_p_yhf_var')
    default_bmfmc_model.predictive_var_bool = True

    default_bmfmc_model.compute_pyhf_statistics()
    mp1.assert_called_once()
    mp2.assert_called_once()

    default_bmfmc_model.predictive_var_bool = False
    default_bmfmc_model.compute_pyhf_statistics()
    assert default_bmfmc_model.p_yhf_var is None


@pytest.mark.unit_tests
def test_calculate_p_yhf_mean(default_bmfmc_model):
    default_bmfmc_model.var_y_mc = np.ones((10, 1))
    default_bmfmc_model.y_pdf_support = np.linspace(-1.0, 1.0, 10)
    default_bmfmc_model.m_f_mc = np.linspace(0, 10.0, 10)

    default_bmfmc_model._calculate_p_yhf_mean()

    expected_mean_pdf = np.array(
        [
            2.41970725e-01,
            6.70116261e-02,
            8.42153448e-03,
            4.80270652e-04,
            1.24289548e-05,
            1.45961027e-07,
            7.77844807e-10,
            1.88106042e-12,
            2.06426893e-15,
            1.02797736e-18,
        ]
    )
    np.testing.assert_array_almost_equal(
        default_bmfmc_model.p_yhf_mean, expected_mean_pdf, decimal=9
    )


@pytest.mark.unit_tests
def test_calculate_p_yhf_var(mocker, default_bmfmc_model):
    np.random.seed(1)
    K = np.random.rand(10, 10)
    mp1 = mocker.patch(
        'pqueens.interfaces.bmfmc_interface.BmfmcInterface.map', return_value=(None, K)
    )
    default_bmfmc_model.var_y_mc = np.ones((10, 1))
    default_bmfmc_model.y_pdf_support = np.linspace(-1.0, 1.0, 10)
    default_bmfmc_model.m_f_mc = np.atleast_2d(np.linspace(0, 10.0, 10)).T
    default_bmfmc_model.p_yhf_mean = np.array(
        [
            2.41970725e-01,
            6.70116261e-02,
            8.42153448e-03,
            4.80270652e-04,
            1.24289548e-05,
            1.45961027e-07,
            7.77844807e-10,
            1.88106042e-12,
            2.06426893e-15,
            1.02797736e-18,
        ]
    )

    default_bmfmc_model._calculate_p_yhf_var()

    expected_var = np.array(
        [
            -0.0580733,
            -0.00377325,
            0.00100094,
            0.001509,
            0.00200187,
            0.00251098,
            0.00299406,
            0.00341733,
            0.00376561,
            0.00404483,
        ]
    )
    # asserts / tests
    mp1.assert_called_once()
    np.testing.assert_array_almost_equal(default_bmfmc_model.p_yhf_var, expected_var, decimal=8)


@pytest.mark.unit_tests
def test_compute_pymc_reference(mocker, default_bmfmc_model):
    mp1 = mocker.patch('pqueens.utils.pdf_estimation.estimate_bandwidth_for_kde', return_value=1.0)
    mp2 = mocker.patch('pqueens.utils.pdf_estimation.estimate_pdf', return_value=(1.0, None))

    default_bmfmc_model.Y_LFs_train = np.array([[1.0, 1.0]])
    default_bmfmc_model.compute_pymc_reference()

    # tests / asserts
    mp1.assert_called_once()
    mp2.assert_called_once()


@pytest.mark.unit_tests
def test_set_feature_strategy(mocker, default_bmfmc_model):
    mp1 = mocker.patch('pqueens.models.bmfmc_model.update_model_variables')
    mp2 = mocker.patch(
        'pqueens.models.bmfmc_model.BMFMCModel.update_probabilistic_mapping_with_features'
    )

    np.random.seed(2)
    default_bmfmc_model.gammas_ext_mc = np.random.random((20, 2))

    # test man_features
    default_bmfmc_model.settings_probab_mapping['features_config'] = "man_features"
    default_bmfmc_model.X_train = np.array([[1.0, 1.0], [2.0, 2.0]])
    default_bmfmc_model.Y_LFs_train = np.array([[1.2, 1.2]]).T
    default_bmfmc_model.set_feature_strategy()
    expected_Z_train = np.array([[1.2, 1.2], [1.0, 2.0]]).T
    np.random.seed(1)
    expected_gamma_mc = np.atleast_2d(np.random.random((20, 2))[:, 1]).T
    np.random.seed(2)
    expected_LF_mc = np.random.random((20, 2))
    expected_Z_mc = np.hstack((expected_LF_mc, expected_gamma_mc))

    np.testing.assert_array_almost_equal(default_bmfmc_model.Z_train, expected_Z_train, decimal=6)
    np.testing.assert_array_almost_equal(default_bmfmc_model.Z_mc, expected_Z_mc, decimal=6)

    # test opt_features
    default_bmfmc_model.settings_probab_mapping['features_config'] = "opt_features"
    default_bmfmc_model.set_feature_strategy()
    mp2.assert_called_once()

    default_bmfmc_model.settings_probab_mapping['num_features'] = 0
    with pytest.raises(ValueError):
        default_bmfmc_model.set_feature_strategy()

    # test no_features
    default_bmfmc_model.settings_probab_mapping['features_config'] = "no_features"
    default_bmfmc_model.set_feature_strategy()
    np.testing.assert_array_almost_equal(
        default_bmfmc_model.Z_train, default_bmfmc_model.Y_LFs_train, decimal=6
    )
    np.testing.assert_array_almost_equal(
        default_bmfmc_model.Z_mc, default_bmfmc_model.Y_LFs_mc, decimal=6
    )

    # test the variable update for all cases
    assert mp1.call_count == 3


@pytest.mark.unit_tests
def test_calculate_extended_gammas(mocker, default_bmfmc_model):
    np.random.seed(2)
    y_LFS_mc_stdized = np.random.random((20, 1))

    np.random.seed(1)
    x_red = np.hstack((np.random.random((20, 1)), y_LFS_mc_stdized))

    mp1 = mocker.patch('pqueens.models.bmfmc_model.BMFMCModel.input_dim_red', return_value=x_red)
    mocker.patch(
        'pqueens.visualization.bmfmc_visualization.bmfmc_visualization_instance',
        return_value=mock_visualization,
    )

    mp2 = mocker.patch(
        'pqueens.visualization.bmfmc_visualization.bmfmc_visualization_instance'
        '.plot_feature_ranking'
    )
    mp3 = mocker.patch(
        'pqueens.models.bmfmc_model.StandardScaler.fit_transform', return_value=y_LFS_mc_stdized
    )

    def linear_scale_dummy(a, b):
        return a

    with patch.object(bmfmc_model, '_linear_scale_a_to_b', linear_scale_dummy):
        default_bmfmc_model.calculate_extended_gammas()

        mp1.assert_called_once()
        mp2.assert_called_once()
        mp3.assert_called_once()

        # test scores ranking functionality of informative features
        np.random.seed(1)
        expected_gamma_ext_mc = np.hstack((y_LFS_mc_stdized, np.random.random((20, 1))))
        np.testing.assert_array_almost_equal(
            default_bmfmc_model.gammas_ext_mc, expected_gamma_ext_mc, decimal=6
        )


@pytest.mark.unit_tests
def test_update_probabilistic_mapping_with_features(mocker, default_bmfmc_model):
    mp1 = mocker.patch('pqueens.interfaces.bmfmc_interface.BmfmcInterface.build_approximation')
    mp2 = mocker.patch(
        'pqueens.interfaces.bmfmc_interface.BmfmcInterface.map',
        return_value=(np.array([1.0, 1.1]), np.array([2.0, 2.1])),
    )

    default_bmfmc_model.gammas_ext_mc = np.random.random((20, 5))
    default_bmfmc_model.settings_probab_mapping['num_features'] = 1
    default_bmfmc_model.Y_LFs_mc = np.random.random((20, 1))
    default_bmfmc_model.training_indices = 1

    default_bmfmc_model.update_probabilistic_mapping_with_features()

    mp1.assert_called_once()
    assert mp2.call_count == 2
    assert default_bmfmc_model.Z_mc.shape[1] == 2


@pytest.mark.unit_tests
def test_input_dim_red(mocker, default_bmfmc_model):
    mp1 = mocker.patch(
        'pqueens.models.bmfmc_model.BMFMCModel.get_random_fields_and_truncated_basis',
        return_value=(1.0, 1.0),
    )
    mp2 = mocker.patch(
        'pqueens.models.bmfmc_model._project_samples_on_truncated_basis', return_value=1.0
    )
    mp3 = mocker.patch('pqueens.models.bmfmc_model._assemble_x_red_stdizd', retrun_value=1.0)
    default_bmfmc_model.input_dim_red()

    mp1.assert_called_once()
    mp2.assert_called_once()
    mp3.assert_called_once()


@pytest.mark.unit_tests
def test_get_random_fields_and_truncated_basis(default_bmfmc_model):

    np.random.seed(1)
    random_field = np.random.random((10, 10))
    default_bmfmc_model.eigenfunc_random_fields = {'random_inflow': random_field}
    # note that this is the percentage cummulated amount of the eigenvalue
    default_bmfmc_model.eigenvals = {'random_inflow': np.linspace(1, 100, 10)}
    np.random.seed(1)
    default_bmfmc_model.X_mc = np.hstack((np.random.random((10, 3)), random_field))
    expected_x_uncorr = default_bmfmc_model.X_mc[:, 0:2]
    expected_random_fields_trunc_dict = {'random_inflow': {'samples': random_field[0:8, :]}}

    x_uncorr, random_fields_trunc_dict = default_bmfmc_model.get_random_fields_and_truncated_basis(
        explained_var=80.0
    )

    np.testing.assert_array_almost_equal(
        random_fields_trunc_dict['random_inflow']['trunc_basis'],
        expected_random_fields_trunc_dict['random_inflow']['samples'],
        decimal=6,
    )
    np.testing.assert_array_almost_equal(x_uncorr, expected_x_uncorr, decimal=6)

    # test error catch for other corrstructs in random fields
    default_bmfmc_model.uncertain_parameters['random_fields']['random_inflow'][
        'corrstruct'
    ] = 'dummy'

    with pytest.raises(NotImplementedError):
        default_bmfmc_model.get_random_fields_and_truncated_basis(explained_var=95.0)


@pytest.mark.unit_tests
def test_project_samples_on_truncated_basis():
    np.random.seed(1)
    num_samples = 5
    dim = 4
    red_dim = 2
    samples = np.random.random((num_samples, dim))
    trunc_basis = np.random.random((red_dim, dim))
    truncated_basis_dict = {'random_inflow': {'samples': samples, 'trunc_basis': trunc_basis}}
    expected_coef_mat = np.array(
        [
            [1.24073816, 1.02169791],
            [0.50453986, 0.24055815],
            [1.44520359, 0.89216292],
            [1.48672516, 0.99326356],
            [1.05626325, 0.88520495],
        ]
    )

    coef_mat = bmfmc_model._project_samples_on_truncated_basis(truncated_basis_dict, num_samples)

    np.testing.assert_array_almost_equal(coef_mat, expected_coef_mat, decimal=6)


@pytest.mark.unit_tests
def test_update_model_variables(default_bmfmc_model):
    # np.random.seed(1)
    # Z_LFs_train = np.random.random((5, 2))
    # np.random.seed(2)
    # Z_mc = np.random.random((10, 2))
    # expected_variables = 1

    # bmfmc_model.update_model_variables(Z_LFs_train, Z_mc)
    # TODO method and test do not seem to have any effect for now
    pass


@pytest.mark.unit_tests
def test_linear_scale_a_to_b():
    a_vec = np.linspace(0.0, 1.0, 10)
    b_vec = np.linspace(-5.0, 5.0, 10)

    expected_scaled_a_vec = np.linspace(-5.0, 5.0, 10)
    scaled_a_vec = bmfmc_model._linear_scale_a_to_b(a_vec, b_vec)
    np.testing.assert_array_almost_equal(scaled_a_vec, expected_scaled_a_vec, decimal=6)


@pytest.mark.unit_tests
def test_assemble_x_red_stdizd():
    x_uncorr = np.atleast_2d(np.linspace(0.0, 10.0, 5)).T
    coef_mat = x_uncorr
    expected_X_stdized = np.array(
        [
            [-1.41421356, -1.41421356],
            [-0.70710678, -0.70710678],
            [0.0, 0.0],
            [0.70710678, 0.70710678],
            [1.41421356, 1.41421356],
        ]
    )

    X_stdized = bmfmc_model._assemble_x_red_stdizd(x_uncorr, coef_mat)
    np.testing.assert_array_almost_equal(X_stdized, expected_X_stdized, decimal=6)
