import os
import pickle
import numpy as np

from pqueens.main import main
from pqueens.tests.integration_tests.example_simulator_functions.sinus_test_fun import (
    sinus_test_fun,
)
from pqueens.tests.integration_tests.example_simulator_functions.branin_hifi import branin_hifi
from pqueens.utils import injector


def test_gp_precompiled_one_dim(inputdir, tmpdir):
    """ Test case for GPPrecompiled based GP model """

    template = os.path.join(inputdir, 'gp_precompiled_template.json')
    input_file = os.path.join(tmpdir, 'gp_precompiled.json')

    # pylint: disable=line-too-long
    dir_dict = {
        'test_fun': 'sinus_test_fun.py',
        'variables': '"x1": {"type": "FLOAT","size": 1,"min": -5,"max": 5,"distribution": "uniform","distribution_parameter": [-5,5]}',
    }
    # pylint: enable=line-too-long

    injector.inject(dir_dict, template, input_file)

    arguments = [
        '--input=' + input_file,
        '--output=' + str(tmpdir),
    ]

    main(arguments)
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
    """ Test case for GPPrecompiled based GP model """
    template = os.path.join(inputdir, 'gp_precompiled_template.json')
    input_file = os.path.join(tmpdir, 'gp_precompiled.json')

    # pylint: disable=line-too-long
    dir_dict = {
        'test_fun': 'branin_hifi.py',
        'variables': '"x1": {"type": "FLOAT","size": 1,"min": -5,"max": 10,"distribution": "uniform","distribution_parameter": [-5,10]}, "x2": {"type": "FLOAT","size": 1,"min": 0,"max": 15,"distribution": "uniform","distribution_parameter": [0,15]}',
    }
    # pylint: enable=line-too-long

    injector.inject(dir_dict, template, input_file)

    arguments = [
        '--input=' + input_file,
        '--output=' + str(tmpdir),
    ]

    main(arguments)
    result_file = str(tmpdir) + '/' + 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    # evaluate the testing/benchmark function at testing inputs
    x1_vec = results['raw_output_data']['x_test'][:, 0]
    x2_vec = results['raw_output_data']['x_test'][:, 1]
    fun_mat = branin_hifi(x1_vec, x2_vec)
    var_mat = np.zeros(fun_mat.shape)

    np.testing.assert_array_almost_equal(
        results["raw_output_data"]["mean"].flatten(), fun_mat, decimal=-1
    )

    np.testing.assert_array_almost_equal(
        np.sqrt(results["raw_output_data"]["variance"].flatten()) / 20, var_mat, decimal=0
    )
