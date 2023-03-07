"""Integration tests for the BMFIA."""

import os
import pickle
from pathlib import Path

import numpy as np
import pytest

import pqueens.visualization.bmfia_visualization as qvis
from pqueens import run
from pqueens.utils import injector


def test_smc_park_hf(
    inputdir,
    tmpdir,
    create_experimental_data_park91a_hifi_on_grid,
    expected_samples,
    expected_weights,
):
    """Integration test for BMFIA.

    Integration test for the bayesian multi-fidelity inverse analysis
    (bmfia) using the *park91* function.
    """
    # generate json input file from template
    template = os.path.join(inputdir, 'bmfia_smc_park.yml')
    experimental_data_path = tmpdir
    dir_dict = {"experimental_data_path": experimental_data_path, "plot_dir": tmpdir}
    input_file = os.path.join(tmpdir, 'smc_mf_park_realization.yml')
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
            [0.4668699, 0.77473038],
            [0.49999182, 0.77607136],
            [0.50663332, 0.77550401],
            [0.49516163, 0.97392396],
            [0.31519886, 0.17738099],
            [0.40139626, 0.43184985],
            [0.28054033, 0.32720431],
            [0.48645546, 0.79040136],
            [0.83723558, 0.72781892],
            [0.60518726, 0.66127113],
            [0.84825265, 0.85108898],
            [0.53457146, 0.79481293],
            [0.60946442, 0.90369187],
            [0.51133919, 0.84280775],
            [0.67083503, 0.72895975],
            [0.53378982, 0.75098966],
            [0.45745844, 0.79150874],
            [0.83389204, 0.79212691],
            [0.80604362, 0.84595597],
            [0.87414066, 0.86018478],
            [0.64688654, 0.92045896],
            [0.76267715, 0.89186875],
            [0.56441915, 0.69882997],
            [0.71478118, 0.97678315],
            [0.45189243, 0.75833152],
            [0.50174979, 0.66130005],
            [0.5823038, 0.70649317],
            [0.51445655, 0.85663335],
            [0.58403945, 0.3389402],
            [0.58403945, 0.3389402],
            [0.57860791, 0.43456372],
            [0.47624353, 0.44104932],
            [0.52983446, 0.75417905],
            [0.62055361, 0.65612744],
            [0.60320455, 0.73685999],
            [0.69545747, 0.89971902],
            [0.43849334, 0.58214074],
            [0.7529578, 0.82887988],
            [0.86750406, 0.77078031],
            [0.73878006, 0.87415752],
            [0.77208488, 0.79098077],
            [0.88521055, 0.81967215],
            [0.94459114, 0.75275776],
            [0.89518535, 0.8219029],
            [0.76590761, 0.83828475],
            [0.66283456, 0.8411014],
            [0.52164303, 0.7414026],
            [0.5405238, 0.72503477],
            [0.69193464, 0.83765331],
            [0.72592338, 0.98350828],
        ]
    )

    return samples


@pytest.fixture()
def expected_weights():
    """Fixture for expected SMC weights."""
    weights = np.array(
        [
            0.06072509,
            0.05991984,
            0.05522222,
            0.03570726,
            0.00010481,
            0.00081085,
            0.00044532,
            0.01950984,
            0.0005873,
            0.00021821,
            0.00241822,
            0.06518743,
            0.03983215,
            0.06164492,
            0.03323568,
            0.03793718,
            0.03164601,
            0.00406821,
            0.00533648,
            0.00333487,
            0.00808975,
            0.00866996,
            0.02188862,
            0.01218727,
            0.03027108,
            0.01837699,
            0.03102733,
            0.02681679,
            0.00650307,
            0.00650307,
            0.00522202,
            0.00556961,
            0.01780405,
            0.02476554,
            0.0365897,
            0.03917174,
            0.02897047,
            0.01129402,
            0.00555813,
            0.01185257,
            0.0114345,
            0.00362188,
            0.00170968,
            0.00331061,
            0.00385532,
            0.0022185,
            0.02003034,
            0.03643343,
            0.01939011,
            0.02297195,
        ]
    )
    return weights
