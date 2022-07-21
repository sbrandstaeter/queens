import os
import pickle
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import entropy

import pqueens.utils.pdf_estimation as est
from pqueens import run
from pqueens.tests.integration_tests.example_simulator_functions.currin88 import (
    currin88_hifi,
    currin88_lofi,
)
from pqueens.utils import injector
from pqueens.utils.process_outputs import write_results


# ---- fixtures ----------------------------------------------------------------
@pytest.fixture()
def generate_X_mc():
    # generate 5000 uniform samples for x1 and x2 in [0,1]
    np.random.seed(1)
    n_samples = 1000
    X_mc = np.random.uniform(low=0.0, high=1.0, size=(n_samples, 2))
    return X_mc


@pytest.fixture()
def generate_LF_MC_data(generate_X_mc):
    y = []
    for x_vec in generate_X_mc:
        params = {'x1': x_vec[0], 'x2': x_vec[1]}
        y.append(currin88_lofi(**params))

    Y_LF_mc = np.array(y).reshape((generate_X_mc.shape[0], -1))

    return Y_LF_mc


@pytest.fixture()
def generate_HF_MC_data(generate_X_mc):
    y = []
    for x_vec in generate_X_mc:
        params = {'x1': x_vec[0], 'x2': x_vec[1]}
        y.append(currin88_hifi(**params))

    Y_LF_mc = np.array(y).reshape((generate_X_mc.shape[0], -1))

    return Y_LF_mc


@pytest.fixture()
def write_LF_MC_data_to_pickle(tmpdir, generate_X_mc, generate_LF_MC_data):
    file_name = 'LF_MC_data'
    input_description = {
        "random_variables": {
            "x1": {
                "dimension": 1,
                "distribution": "uniform",
                "lower_bound": 0.0,
                "upper_bound": 1.0,
            },
            "x2": {
                "dimension": 1,
                "distribution": "uniform",
                "lower_bound": 0.0,
                "upper_bound": 1.0,
            },
        },
        "random_fields": None,
    }
    data = {
        'input_data': generate_X_mc,
        'input_description': input_description,
        'output': generate_LF_MC_data,
        'eigenfunc': None,
        'eigenvalue': None,
    }
    write_results(data, tmpdir, file_name)


@pytest.fixture(params=['random', 'diverse_subset'])
def design_method(request):
    design = request.param
    return design


# ---- actual integration tests ------------------------------------------------
@pytest.mark.integration_tests
def test_bmfmc_iterator_currin88_random_vars_diverse_design(
    tmpdir,
    inputdir,
    write_LF_MC_data_to_pickle,
    generate_HF_MC_data,
    generate_LF_MC_data,
    design_method,
):
    """Integration tests for the BMFMC routine based on the HF and LF currin88
    function."""
    # generate json input file from template
    template = os.path.join(inputdir, 'bmfmc_currin88_template.json')
    plot_dir = tmpdir
    lf_mc_data_name = 'LF_MC_data.pickle'
    path_lf_mc_pickle_file = os.path.join(tmpdir, lf_mc_data_name)
    dir_dict = {
        'lf_mc_pickle_file': path_lf_mc_pickle_file,
        'plot_dir': plot_dir,
        'design_method': design_method,
    }
    input_file = os.path.join(tmpdir, 'bmfmc_currin88.json')
    injector.inject(dir_dict, template, input_file)

    # run the main routine of QUEENS
    run(Path(input_file), Path(tmpdir))

    # actual main call of BMFMC

    # get the results of the QUEENS run
    result_file = os.path.join(tmpdir, 'bmfmc_currin88.pickle')
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    # get the y_support and calculate HF MC reference
    y_pdf_support = results['raw_output_data']['y_pdf_support']
    Y_LFs_mc = generate_LF_MC_data
    Y_HF_mc = generate_HF_MC_data
    bandwidth_lfmc = est.estimate_bandwidth_for_kde(
        Y_LFs_mc[:, 0], np.amin(Y_LFs_mc[:, 0]), np.amax(Y_LFs_mc[:, 0])
    )

    p_yhf_mc, _ = est.estimate_pdf(
        np.atleast_2d(Y_HF_mc).T, bandwidth_lfmc, support_points=np.atleast_2d(y_pdf_support)
    )

    kl_divergence = entropy(p_yhf_mc, results['raw_output_data']['p_yhf_mean'])
    assert kl_divergence < 0.3
