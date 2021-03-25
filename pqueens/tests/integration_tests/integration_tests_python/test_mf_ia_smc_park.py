import os
import pickle
import pytest
import numpy as np
import pandas as pd
from pqueens.main import main
from pqueens.utils import injector
from pqueens.tests.integration_tests.example_simulator_functions.park91a_hifi_coords import (
    park91a_hifi_coords,
)


def test_smc_park_hf(
    inputdir, tmpdir, design_and_write_experimental_data_to_csv, expected_samples, expected_weights
):
    """
    Integration test for bayesian multi-fidelity inverse analysis (bmfia)
    using the park91 function
    """

    # generate json input file from template
    template = os.path.join(inputdir, 'mf_ia_smc_park.json')
    experimental_data_path = tmpdir
    dir_dict = {'experimental_data_path': experimental_data_path}
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

    # some tests / asserts here
    np.testing.assert_array_almost_equal(samples, expected_samples)
    np.testing.assert_array_almost_equal(weights, expected_weights)


@pytest.fixture()
def expected_samples():
    samples = np.array(
        [
            [0.4963235, 0.53956207],
            [0.52163527, -0.33021349],
            [0.50378226, 0.31510838],
            [0.38718656, 0.59682355],
            [0.82005518, 1.13231858],
            [0.43898324, 0.24444579],
            [0.45511732, 0.41337149],
            [0.52163454, 0.49962882],
        ]
    )
    return samples


@pytest.fixture()
def expected_weights():
    weights = np.array(
        [
            0.1549016,
            0.12157362,
            0.10299685,
            0.14721317,
            0.02461221,
            0.15618766,
            0.15332817,
            0.13918671,
        ]
    )
    return weights


@pytest.fixture()
def design_and_write_experimental_data_to_csv(tmpdir):
    # Fix random seed
    np.random.seed(seed=1)

    # create target inputs
    x1 = 0.5
    x2 = 0.2

    # use x3 and x4 as coordinates and create coordinate grid (same as in park91a_hifi_coords)
    xx3 = np.linspace(0.0, 1.0, 4)
    xx4 = np.linspace(0.0, 1.0, 4)
    x3_vec, x4_vec = np.meshgrid(xx3, xx4)
    x3_vec = x3_vec.flatten()
    x4_vec = x4_vec.flatten()

    # generate clean function output for fake test data
    y_vec = []
    for x3, x4 in zip(x3_vec, x4_vec):
        y_vec.append(park91a_hifi_coords(x1, x2, x3, x4))
    y_vec = np.array(y_vec)

    # add artificial noise to fake measurements
    sigma_n = 0.1
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
