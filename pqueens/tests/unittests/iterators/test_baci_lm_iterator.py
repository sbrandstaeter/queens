import numpy as np
import pytest
import pandas as pd
import os
import plotly.express as px

from collections import OrderedDict

from pqueens.iterators.baci_lm_iterator import BaciLMIterator


@pytest.fixture(scope='module', params=[None, 'method'])
def iterator_name_cases(request):
    return request.param


@pytest.fixture(scope='module', params=[None, 'dummy_model'])
def model_cases(request):
    return request.param


@pytest.fixture(scope='module', params=['grad', 'res', 'not_valid'])
def fix_update_reg(request):
    return request.param


@pytest.fixture(scope='module', params=[1e-6, 1e0])
def fix_tolerance(request):
    return request.param


@pytest.fixture()
def default_baci_lm_iterator():
    config = {
        'database': OrderedDict([('address', 'localhost:27017'), ('drop_existing', True)]),
        'method': OrderedDict(
            [
                ('method_name', 'baci_lm'),
                (
                    'method_options',
                    OrderedDict(
                        [
                            ('model', 'model'),
                            ('jac_method', '2-point'),
                            ('jac_rel_step', 1e-05),
                            ('jac_abs_step', 0.001),
                            ('max_feval', 99),
                            ('init_reg', 1.0),
                            ('update_reg', 'grad'),
                            ('convergence_tolerance', 1e-06),
                            ('initial_guess', [0.1, 0.2]),
                            (
                                'result_description',
                                OrderedDict([('write_results', True), ('plot_results', True)]),
                            ),
                        ]
                    ),
                ),
            ]
        ),
        'model': OrderedDict(
            [('type', 'simulation_model'), ('interface', 'interface'), ('parameters', 'parameters')]
        ),
        'interface': OrderedDict(
            [('type', 'direct_python_interface'), ('main_file', 'rosenbrock_residual.py')]
        ),
        'parameters': OrderedDict(
            [
                (
                    'random_variables',
                    OrderedDict(
                        [
                            ('x1', OrderedDict([('type', 'FLOAT'), ('size', 1)])),
                            ('x2', OrderedDict([('type', 'FLOAT'), ('size', 1)])),
                        ]
                    ),
                )
            ]
        ),
        'debug': False,
        'input_file': 'dummy_input',
        'global_settings': {'output_dir': 'dummy_output', 'experiment_name': 'OptimizeLM',},
    }

    baci_lm_i = BaciLMIterator.from_config_create_iterator(config,)

    return baci_lm_i


@pytest.fixture(scope='module', params=[True, False])
def fix_true_false_param(request):
    return request.param


@pytest.fixture(scope='module')
def fix_plotly_fig():
    data = pd.DataFrame({'x': [1.0, 2.0], 'y': [1.1, 2.1], 'z': [1.2, 2.2]})
    fig = px.line_3d(data, x='x', y='y', z='z',)
    return fig


def test_init(mocker):

    global_settings = {'output_dir': 'dummyoutput', 'experiment_name': 'dummy_exp_name'}
    initial_guess = np.array([1, 2.2])
    jac_rel_step = 1e-3
    jac_abs_step = 1e-2
    init_reg = 1.0
    update_reg = 'grad'
    tolerance = 1e-8
    max_feval = 99
    model = 'dummy_model'
    result_description = (True,)
    verbose_output = (True,)

    mp = mocker.patch('pqueens.iterators.iterator.Iterator.__init__')

    my_baci_lm_iterator = BaciLMIterator(
        global_settings,
        initial_guess,
        jac_rel_step,
        jac_abs_step,
        init_reg,
        update_reg,
        tolerance,
        max_feval,
        model,
        result_description,
        verbose_output,
    )

    mp.assert_called_once_with(model, global_settings)

    np.testing.assert_equal(my_baci_lm_iterator.initial_guess, initial_guess)
    np.testing.assert_equal(my_baci_lm_iterator.param_current, initial_guess)
    assert my_baci_lm_iterator.jac_rel_step == jac_rel_step
    assert my_baci_lm_iterator.max_feval == max_feval
    assert my_baci_lm_iterator.result_description == result_description
    assert my_baci_lm_iterator.jac_abs_step == jac_abs_step
    assert my_baci_lm_iterator.reg_param == init_reg
    assert my_baci_lm_iterator.update_reg == update_reg
    assert my_baci_lm_iterator.verbose_output == verbose_output


def test_from_config_create_iterator(mocker, iterator_name_cases, model_cases):

    config = {
        'method': OrderedDict(
            [
                ('method_name', 'baci_lm'),
                (
                    'method_options',
                    OrderedDict(
                        [
                            ('model', 'model'),
                            ('jac_method', '2-point'),
                            ('jac_rel_step', 1e-05),
                            ('jac_abs_step', 0.001),
                            ('max_feval', 99),
                            ('init_reg', 1.0),
                            ('update_reg', 'grad'),
                            ('convergence_tolerance', 1e-06),
                            ('initial_guess', [0.1, 0.2]),
                            (
                                'result_description',
                                OrderedDict([('write_results', True), ('plot_results', True)]),
                            ),
                        ]
                    ),
                ),
            ]
        ),
        'model': OrderedDict(
            [('type', 'simulation_model'), ('interface', 'interface'), ('parameters', 'parameters')]
        ),
        'input_file': 'input_path',
        'global_settings': {'output_dir': 'output_path', 'experiment_name': 'experimentname',},
    }

    mp = mocker.patch(
        'pqueens.models.model.Model.from_config_create_model', return_value='dummy_model'
    )

    mockinit = mocker.patch(
        'pqueens.iterators.baci_lm_iterator.BaciLMIterator.__init__', return_value=None
    )

    my_iterator = BaciLMIterator.from_config_create_iterator(
        config, iterator_name=iterator_name_cases, model=model_cases
    )
    if model_cases == None:
        mp.assert_called_once_with('model', config)

    mockinit.assert_called_once()

    callargs = mockinit.call_args[1]

    assert callargs['global_settings'] == {
        'output_dir': 'output_path',
        'experiment_name': 'experimentname',
    }
    # assert_called_once_with not possible because we need numpy array comparison
    np.testing.assert_equal(callargs['initial_guess'], np.array([0.1, 0.2]))
    assert callargs['jac_rel_step'] == 1e-05
    assert callargs['jac_abs_step'] == 0.001
    assert callargs['init_reg'] == 1.0
    assert callargs['update_reg'] == 'grad'
    assert callargs['tolerance'] == 1e-06
    assert callargs['max_feval'] == 99
    assert callargs['model'] == 'dummy_model'
    assert callargs['result_description'] == OrderedDict(
        [('write_results', True), ('plot_results', True)]
    )
    assert callargs['verbose_output'] == False


def test_eval_model(default_baci_lm_iterator, mocker):

    mp = mocker.patch('pqueens.models.simulation_model.SimulationModel.evaluate', return_value=None)
    default_baci_lm_iterator.eval_model()
    mp.assert_called_once()


def test_residual(default_baci_lm_iterator, fix_true_false_param, mocker):

    m1 = mocker.patch(
        'pqueens.iterators.baci_lm_iterator.BaciLMIterator' '.get_positions_raw_2pointperturb',
        return_value=[np.array([[1.0, 2.2], [1.00101, 2.2], [1.0, 2.201022]]), None],
    )

    m2 = mocker.patch(
        'pqueens.models.simulation_model.SimulationModel'
        '.check_for_precalculated_response_of_sample_batch',
        return_value=fix_true_false_param,
    )

    m3 = mocker.patch(
        'pqueens.models.simulation_model.SimulationModel' '.update_model_from_sample_batch',
        return_value=None,
    )

    m4 = mocker.patch(
        'pqueens.iterators.baci_lm_iterator.BaciLMIterator.eval_model', return_value=None,
    )

    default_baci_lm_iterator.model.response = {'mean': np.array([[3.0, 4.2], [99.9, 99.9]])}

    result = default_baci_lm_iterator.residual(np.array([1.0, 2.2]))

    np.testing.assert_equal(result, np.array([3.0, 4.2]))

    if fix_true_false_param == False:
        m4.assert_called_once()
        m3.assert_called_once()
    else:
        assert not m4.called
        assert not m3.called

    m2.assert_called_once()


def test_jacobian(default_baci_lm_iterator, fix_true_false_param, mocker):
    m1 = mocker.patch(
        'pqueens.iterators.baci_lm_iterator.BaciLMIterator.get_positions_raw_2pointperturb',
        return_value=[
            np.array([[1.0, 2.2], [1.00101, 2.2], [1.0, 2.201022]]),
            np.array([0.00101, 0.201022]),
        ],
    )

    m2 = mocker.patch(
        'pqueens.models.simulation_model.SimulationModel'
        '.check_for_precalculated_response_of_sample_batch',
        return_value=fix_true_false_param,
    )

    m3 = mocker.patch(
        'pqueens.models.simulation_model.SimulationModel.update_model_from_sample_batch',
        return_value=None,
    )

    m4 = mocker.patch(
        'pqueens.iterators.baci_lm_iterator.BaciLMIterator.eval_model', return_value=None,
    )

    default_baci_lm_iterator.model.response = {'mean': np.array([[3.0, 4.2], [99.9, 99.9]])}

    m5 = mocker.patch(
        'pqueens.iterators.baci_lm_iterator.fd_jacobian',
        return_value=np.array([[1.0, 0.0], [0.0, 1.0]]),
    )

    jacobian = default_baci_lm_iterator.jacobian(np.array([1.0, 2.2]))

    if fix_true_false_param == False:
        m4.assert_called_once()
        m3.assert_called_once()
    else:
        assert not m4.called
        assert not m3.called

    m2.assert_called_once()
    m5.assert_called_once()

    np.testing.assert_equal(jacobian, np.array([[1.0, 0.0], [0.0, 1.0]]))

    if fix_true_false_param == True:
        with pytest.raises(ValueError):
            m5.return_value = np.array([[1.1, 2.2]])
            jac = default_baci_lm_iterator.jacobian(np.array([0.1]))


def test_initialize_run(mocker, fix_true_false_param, default_baci_lm_iterator):
    default_baci_lm_iterator.result_description['write_results'] = fix_true_false_param

    m1 = mocker.patch('builtins.open',)
    m2 = mocker.patch('pandas.core.generic.NDFrame.to_csv')

    default_baci_lm_iterator.initialize_run()

    if fix_true_false_param:
        m1.assert_called_once_with(os.path.join('dummy_output', 'OptimizeLM' + '.csv'), 'w')
        m2.assert_called_once_with(m1.return_value.__enter__.return_value, sep='\t', index=None)

    else:
        assert not m1.called
        assert not m2.called
        default_baci_lm_iterator.result_description = None
        default_baci_lm_iterator.initialize_run()


def test_core_run(default_baci_lm_iterator, mocker, fix_update_reg, fix_tolerance):
    m1 = mocker.patch(
        'pqueens.iterators.baci_lm_iterator.BaciLMIterator.jacobian',
        return_value=np.array([[1.0, 2.0], [0.0, 1.0]]),
    )

    m2 = mocker.patch(
        'pqueens.iterators.baci_lm_iterator.BaciLMIterator.residual',
        return_value=np.array([0.1, 0.01]),
    )

    m3 = mocker.patch('pqueens.iterators.baci_lm_iterator.BaciLMIterator.printstep')
    default_baci_lm_iterator.update_reg = fix_update_reg
    default_baci_lm_iterator.max_feval = 2
    default_baci_lm_iterator.tolerance = fix_tolerance

    if fix_update_reg not in ['grad', 'res']:
        with pytest.raises(ValueError):
            default_baci_lm_iterator.core_run()
    else:
        default_baci_lm_iterator.core_run()
        if fix_tolerance == 1.0:
            assert m1.call_count == 1
            assert m2.call_count == 1
            assert m3.call_count == 1
        else:
            assert m1.call_count == 3
            assert m2.call_count == 3
            assert m3.call_count == 3
            np.testing.assert_almost_equal(
                default_baci_lm_iterator.param_opt, np.array([0.0275, 0.1725]), 1e-12
            )


def test_post_run(mocker, default_baci_lm_iterator, fix_true_false_param, fix_plotly_fig):

    default_baci_lm_iterator.solution = np.array([1.1, 2.2])
    default_baci_lm_iterator.iter_opt = 3

    pdata = pd.DataFrame({'params': ['[1.0e3 2.0e-2]', '[1.1 2.1]'], 'resnorm': [1.2, 2.2]})
    checkdata = pd.DataFrame({'resnorm': [1.2, 2.2], 'x1': [1000.0, 1.1], 'x2': [0.02, 2.1]})

    default_baci_lm_iterator.result_description['plot_results'] = fix_true_false_param
    m1 = mocker.patch('pandas.read_csv', return_value=pdata)
    m2 = mocker.patch('plotly.express.line_3d', return_value=fix_plotly_fig)
    m3 = mocker.patch('plotly.basedatatypes.BaseFigure.update_traces', return_value=None)
    m4 = mocker.patch('plotly.basedatatypes.BaseFigure.write_html', return_value=None)

    default_baci_lm_iterator.post_run()

    if fix_true_false_param:
        m1.assert_called_once_with(os.path.join('dummy_output', 'OptimizeLM' + '.csv'), sep='\t')
        callargs = m2.call_args
        pd.testing.assert_frame_equal(callargs[0][0], checkdata)
        assert callargs[1]['x'] == 'x1'
        assert callargs[1]['y'] == 'x2'
        assert callargs[1]['z'] == 'resnorm'
        assert callargs[1]['hover_data'] == [
            'iter',
            'resnorm',
            'gradnorm',
            'delta_params',
            'mu',
            'x1',
            'x2',
        ]
        m4.assert_called_once_with(os.path.join('dummy_output', 'OptimizeLM' + '.html'))
        default_baci_lm_iterator.model.variables[0].variables['x3'] = {
            'size': 1,
            'value': [None],
            'type': 'FLOAT',
            'distribution': None,
            'active': True,
        }
        m5 = mocker.patch('builtins.print')
        m1.return_value = pd.DataFrame(
            {'params': ['[1.0e3 2.0e-2 3.]', '[1.1 2.1 3.1]'], 'resnorm': [1.2, 2.2]}
        )
        default_baci_lm_iterator.post_run()
        m5.assert_called_with(
            'write_results for more than 2 parameters not implemented, because we are limited '
            + 'to 3 dimensions. You have: 3.'
        )
    else:
        default_baci_lm_iterator.result_description = None
        default_baci_lm_iterator.post_run()
        m1.assert_not_called()
        m2.assert_not_called()
        m3.assert_not_called()
        m4.assert_not_called()


def test_get_positions_raw_2pointperturb(default_baci_lm_iterator):
    x = np.array([1.1, 2.5])
    pos, delta_pos = default_baci_lm_iterator.get_positions_raw_2pointperturb(x)
    np.testing.assert_almost_equal(
        pos, np.array([[1.1, 2.5], [1.101011, 2.5], [1.1, 2.501025]]), 1e-12
    )
    np.testing.assert_almost_equal(delta_pos, np.array([[0.001011], [0.001025]]), 1e-12)


def test_printstep(mocker, default_baci_lm_iterator, fix_true_false_param):
    default_baci_lm_iterator.result_description['write_results'] = fix_true_false_param

    m1 = mocker.patch('builtins.open',)
    m2 = mocker.patch('pandas.core.generic.NDFrame.to_csv')

    default_baci_lm_iterator.printstep(5, 1e-3, 1e-4, np.array([10.1, 11.2]))

    if fix_true_false_param:
        m1.assert_called_once_with(os.path.join('dummy_output', 'OptimizeLM' + '.csv'), 'a')
        m2.assert_called_once_with(
            m1.return_value.__enter__.return_value,
            sep='\t',
            header=None,
            mode='a',
            index=None,
            float_format='%.6f',
        )

    else:
        assert not m1.called
        assert not m2.called
        default_baci_lm_iterator.result_description = None
        default_baci_lm_iterator.printstep(5, 1e-3, 1e-4, np.array([10.1, 11.2]))