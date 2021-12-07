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
    """Integration test for bayesian multi-fidelity inverse analysis (bmfia)
    using the park91 function."""

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
    samples = np.array(
        [
            [0.46888022, 0.72378392],
            [0.3962766, 0.56112657],
            [0.40576547, 0.70354832],
            [0.46850841, 0.01965466],
            [0.43861299, 0.76440247],
            [0.49087226, 0.05372344],
            [0.52799981, 0.2385079],
            [0.49666344, 0.26263956],
            [0.32740038, 0.83210069],
            [0.41989239, 0.55248636],
            [0.54576435, 0.20042237],
            [0.43979497, 0.47518611],
            [0.36348031, 0.8657442],
            [0.41852601, 0.48905452],
            [0.43099717, 0.2954069],
            [0.49502586, 0.35327539],
            [0.46786024, 0.30772932],
            [0.41144255, 0.13534946],
            [0.46185785, 0.54836681],
            [0.46895928, 0.68271863],
            [0.493066, 0.02290472],
            [0.54481772, 0.48215798],
            [0.52830975, 0.39993871],
            [0.39164457, 0.85154858],
            [0.47517702, 0.5181734],
            [0.37382446, 0.79583439],
            [0.43046736, 0.72781887],
            [0.31783731, 0.95333176],
            [0.35859121, 0.79771517],
            [0.35747865, 0.67097514],
            [0.39746243, 0.78506694],
            [0.3972286, 0.6828397],
            [0.44220651, 0.12299294],
            [0.37462781, 0.97324616],
            [0.46972003, 0.72164002],
            [0.51088803, 0.31081237],
            [0.4461656, 0.66179728],
            [0.4137127, 0.71323328],
            [0.43143851, 0.30245857],
            [0.50816028, 0.59602645],
            [0.43287032, 0.14452957],
            [0.44356002, 0.43041849],
            [0.49074039, 0.23332494],
            [0.43163041, 0.04374888],
            [0.30880294, 0.84993923],
            [0.52005237, 0.54003269],
            [0.49452955, 0.40139667],
            [0.44431765, 0.48583235],
            [0.51708321, 0.29649355],
            [0.39456413, 0.77828355],
        ]
    )

    return samples


@pytest.fixture()
def expected_weights():
    weights = np.array(
        [
            [0.02047435],
            [0.02036006],
            [0.02023923],
            [0.02040986],
            [0.02014289],
            [0.02064204],
            [0.02053119],
            [0.0204904],
            [0.01973789],
            [0.01951835],
            [0.01964255],
            [0.01945239],
            [0.02043908],
            [0.02045968],
            [0.01998797],
            [0.02051898],
            [0.01990526],
            [0.01928782],
            [0.02047349],
            [0.02058203],
            [0.0206801],
            [0.01765776],
            [0.0173466],
            [0.01952817],
            [0.02047083],
            [0.02008091],
            [0.01964484],
            [0.02029938],
            [0.02021957],
            [0.02022301],
            [0.0201662],
            [0.01986519],
            [0.02037048],
            [0.02047193],
            [0.01949852],
            [0.01994035],
            [0.01964284],
            [0.02057932],
            [0.02047211],
            [0.0199229],
            [0.02041919],
            [0.02036432],
            [0.02041775],
            [0.01960658],
            [0.01890513],
            [0.019289],
            [0.01974189],
            [0.02019899],
            [0.0204882],
            [0.02019242],
        ]
    )
    return weights


@pytest.fixture()
def design_and_write_experimental_data_to_csv(tmpdir):
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
