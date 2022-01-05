"""Integration tests for the BMFIA."""

import os
import pickle

import numpy as np
import pandas as pd
import pytest

import pqueens.visualization.bmfia_visualization as qvis
from pqueens.main import main
from pqueens.tests.integration_tests.example_simulator_functions.park91a_hifi_coords import (
    park91a_hifi_coords,
)
from pqueens.utils import injector


@pytest.mark.integration_tests
def test_smc_park_hf(
    inputdir, tmpdir, design_and_write_experimental_data_to_csv, expected_samples, expected_weights
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
    arguments = [
        '--input=' + input_file,
        '--output=' + str(tmpdir),
    ]

    # actual main call of smc
    main(arguments)

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
    np.testing.assert_array_almost_equal(samples, expected_samples)
    np.testing.assert_array_almost_equal(weights.flatten(), expected_weights.flatten())


@pytest.fixture()
def expected_samples():
    """Fixture for expected SMC samples."""
    samples = np.array(
        [
            [0.499052, 0.70257053],
            [0.39858925, 0.6259848],
            [0.42076947, 0.79058332],
            [0.44906662, 0.27293625],
            [0.38491611, 0.86924963],
            [0.46532688, 0.06462709],
            [0.64408572, 0.24145831],
            [0.35784139, 0.85183711],
            [0.26642146, 0.91385232],
            [0.53248334, 0.35699063],
            [0.5034537, 0.25572163],
            [0.4642543, 0.38673243],
            [0.50892306, 0.38538954],
            [0.56742564, 0.12654127],
            [0.27558199, 0.87406743],
            [0.54723426, 0.29318641],
            [0.49751668, 0.6075689],
            [0.48328293, 0.11311099],
            [0.48488195, 0.36906814],
            [0.52747971, 0.48552484],
            [0.53949103, 0.0991292],
            [0.46206128, 0.65071388],
            [0.47505699, 0.33373728],
            [0.445633, 0.81654037],
            [0.54851388, 0.32667545],
            [0.46306658, 0.46474513],
            [0.49145419, 0.61697175],
            [0.34543029, 0.94801499],
            [0.46034, 0.76653459],
            [0.36957172, 0.60856599],
            [0.391956, 0.79489078],
            [0.47564755, 0.54887079],
            [0.4724102, 0.11277707],
            [0.36150221, 0.84978995],
            [0.48490743, 0.45529966],
            [0.50590148, 0.56125645],
            [0.44057981, 0.530901],
            [0.45366165, 0.63667073],
            [0.36734547, 0.7946257],
            [0.50379619, 0.7122354],
            [0.40486123, 0.48679493],
            [0.49394072, 0.44072495],
            [0.57991323, 0.1914979],
            [0.56101678, 0.05405998],
            [0.37517823, 0.53941247],
            [0.63620931, 0.27279017],
            [0.53196536, 0.25414007],
            [0.47421928, 0.32977136],
            [0.50063777, 0.29115533],
            [0.41160247, 0.73937494],
        ]
    )

    return samples


@pytest.fixture()
def expected_weights():
    """Fixture for expecte SMC weights."""
    weights = np.array(
        [
            0.02085062,
            0.02067192,
            0.020239,
            0.01905444,
            0.02091601,
            0.01982684,
            0.01730117,
            0.01847098,
            0.01834809,
            0.02017424,
            0.02059079,
            0.01854839,
            0.0208477,
            0.02067417,
            0.01984324,
            0.02081323,
            0.01995698,
            0.01918552,
            0.02037955,
            0.02069791,
            0.02072056,
            0.0207343,
            0.02042903,
            0.01980473,
            0.02048397,
            0.0199419,
            0.01860568,
            0.02055442,
            0.01830043,
            0.02045326,
            0.01975469,
            0.02051372,
            0.02012805,
            0.02061019,
            0.02016302,
            0.02050027,
            0.02095531,
            0.02089292,
            0.02061243,
            0.02057426,
            0.01929027,
            0.0209364,
            0.02039949,
            0.01955977,
            0.01803566,
            0.01826649,
            0.02066626,
            0.0206603,
            0.01953406,
            0.02052737,
        ]
    )
    return weights


@pytest.fixture()
def design_and_write_experimental_data_to_csv(tmpdir):
    """Fixture for generation of experimental data."""
    # Fix random seed
    np.random.seed(seed=1)

    # create target inputs
    x1 = 0.5
    x2 = 0.1

    # use x3 and x4 as coordinates and create coordinate grid (same as in park91a_hifi_coords)
    xx3 = np.linspace(0.015, 0.95, 4)
    xx4 = np.linspace(0.015, 0.95, 4)
    x3_vec, x4_vec = np.meshgrid(xx3, xx4)
    x3_vec = x3_vec.flatten()
    x4_vec = x4_vec.flatten()

    # generate clean function output for fake test data
    y_vec = []
    for x3, x4 in zip(x3_vec, x4_vec):
        y_vec.append(park91a_hifi_coords(x1, x2, x3, x4))
    y_vec = np.array(y_vec)

    # add artificial noise to fake measurements
    sigma_n = 0.01
    noise_vec = np.random.normal(loc=0, scale=sigma_n, size=(y_vec.size,))
    y_fake = y_vec + noise_vec

    # write fake data to csv
    data_dict = {
        'x3': x3_vec,
        'x4': x4_vec,
        'y_obs': y_fake,
    }
    experimental_data_path = os.path.join(tmpdir, 'experimental_data.csv')
    df = pd.DataFrame.from_dict(data_dict)
    df.to_csv(experimental_data_path, index=False)
