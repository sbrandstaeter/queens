import numpy as np
import pytest
import pandas as pd

from collections import OrderedDict

import pqueens.iterators.baci_lm_iterator

from pqueens.iterators.baci_lm_iterator import BaciLMIterator


@pytest.fixture(scope='module', params=[None, 'method'])
def iterator_name_cases(request):
    return request.param


@pytest.fixture(scope='module', params=[None, 'dummy_model'])
def model_cases(request):
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
        'global_settings': {
            'output_dir': 'dummy_output',
            'experiment_name': 'OptimizeLM',
        },
    }

    baci_lm_i = BaciLMIterator.from_config_create_iterator(config,)

    return baci_lm_i


@pytest.fixture(scope='module', params=[True, False])
def fix_true_false_param(request):
    return request.param


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
        'pqueens.models.simulation_model.SimulationModel'
        '.update_model_from_sample_batch',
        return_value=None,
    )

    m4 = mocker.patch(
        'pqueens.iterators.baci_lm_iterator.BaciLMIterator.eval_model',
        return_value=None,
    )

    m5 = mocker.patch(
        'numpy.atleast_1d',
        return_value=np.array([3., 4.2]),
    )

    pass
