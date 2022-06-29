"""Integration tests for the BMFIA."""

import os
import pickle

import numpy as np
import pandas as pd
import pytest

import pqueens.visualization.bmfia_visualization as qvis
from pqueens.main import main
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
            [0.50570255, 0.78748257],
            [0.45305331, 0.61440264],
            [0.45002158, 0.88891107],
            [0.56791811, 0.19177197],
            [0.53042589, 0.81449974],
            [0.49065355, 0.6658552],
            [0.50011179, 0.54464128],
            [0.59143006, 0.46430805],
            [0.44967965, 0.824535],
            [0.58677661, 0.64097824],
            [0.72971848, 0.14634572],
            [0.56407034, 0.04227288],
            [0.57953753, 0.45030426],
            [0.61289571, 0.18704362],
            [0.40056366, 0.85616452],
            [0.65374306, 0.30852402],
            [0.64558252, 0.3339256],
            [0.51799006, 0.3089308],
            [0.57927996, 0.42801774],
            [0.59126059, 0.47488297],
            [0.63929769, 0.16530918],
            [0.61491528, 0.32393913],
            [0.5120309, 0.62925829],
            [0.48507435, 0.90013568],
            [0.58561275, 0.63789521],
            [0.60834125, 0.30362592],
            [0.63788195, 0.20806519],
            [0.50874171, 0.80589884],
            [0.6060504, 0.73838468],
            [0.58344188, 0.25655834],
            [0.49000208, 0.74550544],
            [0.29934568, 0.98878081],
            [0.40366089, 0.7616558],
            [0.55598575, 0.72039299],
            [0.41144714, 0.86504722],
            [0.61496879, 0.6813067],
            [0.52217323, 0.76757869],
            [0.40989358, 0.7986126],
            [0.40745666, 0.8310776],
            [0.56629388, 0.68011693],
            [0.44599301, 0.77655088],
            [0.49066814, 0.82469821],
            [0.74057841, 0.24075973],
            [0.61423051, 0.05718505],
            [0.54371153, 0.55758288],
            [0.68587395, 0.36337297],
            [0.59295875, 0.42020199],
            [0.4707681, 0.45634122],
            [0.61203477, 0.29529456],
            [0.43288568, 0.96691316],
        ]
    )

    return samples


@pytest.fixture()
def expected_weights():
    """Fixture for expecte SMC weights."""
    weights = np.array(
        [
            0.02336235,
            0.01761432,
            0.02146324,
            0.02077159,
            0.02027751,
            0.02362222,
            0.02255788,
            0.02328196,
            0.02246376,
            0.02126886,
            0.01414318,
            0.01369267,
            0.02475659,
            0.02180891,
            0.01991189,
            0.024545,
            0.02234056,
            0.01287832,
            0.0249466,
            0.02266059,
            0.01892666,
            0.02325979,
            0.02267479,
            0.01832863,
            0.02204168,
            0.01898965,
            0.01586733,
            0.01937827,
            0.00900045,
            0.02161179,
            0.02146546,
            0.01201253,
            0.01525439,
            0.02260829,
            0.02173507,
            0.01598624,
            0.02050526,
            0.01938848,
            0.01871452,
            0.02189578,
            0.02415992,
            0.02065628,
            0.01102741,
            0.0238959,
            0.0224536,
            0.01902431,
            0.02432446,
            0.01568589,
            0.02271612,
            0.01804304,
        ]
    )
    return weights
