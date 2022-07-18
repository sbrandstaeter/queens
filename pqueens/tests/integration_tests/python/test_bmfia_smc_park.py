"""Integration tests for the BMFIA."""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pqueens.visualization.bmfia_visualization as qvis
from pqueens import run
from pqueens.utils import injector


@pytest.mark.integration_tests
def test_smc_park_hf(
    inputdir,
    tmpdir,
    create_experimental_data_park91a_hifi_on_grid,
    expected_samples,
    expected_weights,
):
    """Integration test for BMFIA.

    Integration test for bayesian multi-fidelity inverse analysis
    (bmfia) using the park91 function.
    """
    # generate json input file from template
    template = os.path.join(inputdir, 'bmfia_smc_park.json')
    experimental_data_path = tmpdir
    dir_dict = {"experimental_data_path": experimental_data_path, "plot_dir": tmpdir}
    input_file = os.path.join(tmpdir, 'smc_mf_park_realization.json')
    injector.inject(dir_dict, template, input_file)

    # run the main routine of QUEENS
    run(Path(input_file), Path(tmpdir))

    # actual main call of smc

    # get the results of the QUEENS run
    result_file = os.path.join(tmpdir, 'smc_park_mf.pickle')
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    samples = results['raw_output_data']['particles'].squeeze()
    weights = results['raw_output_data']['weights'].squeeze()

    # ------------------ to be deleted -------
    dim_labels_lst = ['x_s', 'y_s']
    qvis.bmfia_visualization_instance.plot_posterior_from_samples(samples, weights, dim_labels_lst)
    # ----------------------------------------

    # some tests / asserts here
    np.testing.assert_array_almost_equal(samples, expected_samples, decimal=5)
    np.testing.assert_array_almost_equal(weights.flatten(), expected_weights.flatten(), decimal=5)


@pytest.fixture()
def expected_samples():
    """Fixture for expected SMC samples."""
    samples = np.array(
        [
            [0.61768, 0.53351],
            [0.47577, 0.70264],
            [0.49538, 0.75097],
            [0.54453, 0.61119],
            [0.52891, 0.76951],
            [0.61962, 0.09870],
            [0.66577, 0.25289],
            [0.59683, 0.15198],
            [0.47882, 0.71691],
            [0.63686, 0.50690],
            [0.53242, 0.53348],
            [0.34942, 0.94120],
            [0.45186, 0.80664],
            [0.48562, 0.69987],
            [0.61228, 0.50496],
            [0.61296, 0.37891],
            [0.48725, 0.80754],
            [0.53577, 0.32312],
            [0.54838, 0.55167],
            [0.62882, 0.40447],
            [0.59886, 0.16686],
            [0.64538, 0.42947],
            [0.70653, 0.11993],
            [0.50462, 0.78112],
            [0.64962, 0.37568],
            [0.69012, 0.10732],
            [0.53750, 0.75396],
            [0.41232, 0.98378],
            [0.49493, 0.84181],
            [0.40557, 0.88782],
            [0.49357, 0.69046],
            [0.47855, 0.38372],
            [0.59251, 0.37788],
            [0.59549, 0.63197],
            [0.56551, 0.41982],
            [0.59221, 0.58847],
            [0.45818, 0.70603],
            [0.54406, 0.76282],
            [0.41530, 0.87318],
            [0.49331, 0.78383],
            [0.46907, 0.79717],
            [0.51466, 0.77039],
            [0.63719, 0.22622],
            [0.60068, 0.03029],
            [0.44832, 0.84348],
            [0.65328, 0.39675],
            [0.54733, 0.57694],
            [0.47483, 0.76550],
            [0.68422, 0.11659],
            [0.62280, 0.38360],
        ]
    )

    return samples


@pytest.fixture()
def expected_weights():
    """Fixture for expecte SMC weights."""
    weights = np.array(
        [
            0.01794,
            0.02329,
            0.02813,
            0.02033,
            0.01116,
            0.02335,
            0.01952,
            0.00949,
            0.01160,
            0.00863,
            0.02571,
            0.01223,
            0.03062,
            0.02607,
            0.01700,
            0.02861,
            0.02439,
            0.01062,
            0.02156,
            0.02087,
            0.01255,
            0.01590,
            0.01563,
            0.01060,
            0.02129,
            0.01357,
            0.01770,
            0.02297,
            0.01831,
            0.02260,
            0.02250,
            0.00148,
            0.02154,
            0.02602,
            0.02227,
            0.02301,
            0.02995,
            0.02370,
            0.03005,
            0.02243,
            0.02232,
            0.02150,
            0.02345,
            0.01027,
            0.02618,
            0.01453,
            0.02834,
            0.01902,
            0.02593,
            0.02328,
        ]
    )
    return weights
