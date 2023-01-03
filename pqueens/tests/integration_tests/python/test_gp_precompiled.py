import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from pqueens import run
from pqueens.tests.integration_tests.example_simulator_functions.branin78 import branin78_hifi
from pqueens.tests.integration_tests.example_simulator_functions.sinus import sinus_test_fun
from pqueens.utils import injector


def test_gp_precompiled_one_dim(inputdir, tmpdir):
    """Test case for GPPrecompiled based GP model."""
    template = os.path.join(inputdir, 'gp_precompiled_template.yml')
    input_file = os.path.join(tmpdir, 'gp_precompiled.yml')

    dir_dict = {
        'test_fun': 'sinus_test_fun',
        'variables': {
            "x1": {
                "type": "random_variable",
                "distribution": "uniform",
                "lower_bound": -5,
                "upper_bound": 10,
            }
        },
    }

    injector.inject(dir_dict, template, input_file)

    run(Path(input_file), Path(tmpdir))

    result_file = str(tmpdir) + '/' + 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    # evaluate the testing/benchmark function at testing inputs
    x1_vec = results['raw_output_data']['x_test'][:, 0]
    fun_mat = sinus_test_fun(x1_vec)
    var_mat = np.zeros(fun_mat.shape)

    np.testing.assert_array_almost_equal(
        results["raw_output_data"]["mean"].flatten(), fun_mat, decimal=2
    )
    np.testing.assert_array_almost_equal(
        np.sqrt(results["raw_output_data"]["variance"].flatten()), var_mat, decimal=2
    )


def test_gp_precompiled_two_dim(inputdir, tmpdir):
    """Test case for GPPrecompiled based GP model."""
    template = os.path.join(inputdir, 'gp_precompiled_template.yml')
    input_file = os.path.join(tmpdir, 'gp_precompiled.yml')

    dir_dict = {
        'test_fun': 'branin78_hifi',
        'variables': {
            "x1": {
                "type": "random_variable",
                "distribution": "uniform",
                "lower_bound": -5,
                "upper_bound": 10,
            },
            "x2": {
                "type": "random_variable",
                "distribution": "uniform",
                "lower_bound": 0,
                "upper_bound": 15,
            },
        },
    }

    injector.inject(dir_dict, template, input_file)

    run(Path(input_file), Path(tmpdir))

    result_file = str(tmpdir) + '/' + 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    # evaluate the testing/benchmark function at testing inputs
    x1_vec = results['raw_output_data']['x_test'][:, 0]
    x2_vec = results['raw_output_data']['x_test'][:, 1]
    fun_mat = branin78_hifi(x1_vec, x2_vec)
    var_mat = np.zeros(fun_mat.shape)

    np.testing.assert_array_almost_equal(
        results["raw_output_data"]["mean"].flatten(), fun_mat, decimal=-1
    )

    np.testing.assert_array_almost_equal(
        np.sqrt(results["raw_output_data"]["variance"].flatten()) / 20, var_mat, decimal=0
    )
