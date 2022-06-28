"""Integration test for reparameterization trick VI as executable."""

import os
import pickle

import numpy as np
import pandas as pd
import pytest

from pqueens.main import main
from pqueens.tests.integration_tests.example_simulator_functions.park91a_hifi_coords import (
    park91a_hifi_coords,
)
from pqueens.utils import injector


@pytest.mark.integration_tests_baci
def test_rpvi_iterator_exe_park91a_hifi_provided_gradient(
    inputdir,
    tmpdir,
    design_and_write_experimental_data_to_csv,
    third_party_inputs,
    example_simulator_fun_dir,
):
    """Test for the rpvi iterator based on the park91a_hifi function."""
    # generate json input file from template
    template = os.path.join(inputdir, "rpvi_exe_park91a_hifi_template.json")
    third_party_input_file = os.path.join(
        third_party_inputs, "python_input_files", "input_file_exe_park91a_hifi_coords_gradient.csv"
    )
    experimental_data_path = tmpdir
    executable = os.path.join(example_simulator_fun_dir, "exe_park91a_hifi_coords_gradient.py")
    plot_dir = tmpdir
    dir_dict = {
        "experimental_data_path": experimental_data_path,
        "plot_dir": plot_dir,
        "input_file": third_party_input_file,
        "executable": executable,
        "experiment_dir": tmpdir,
    }
    input_file = os.path.join(tmpdir, "rpvi_park91a_hifi.json")
    injector.inject(dir_dict, template, input_file)

    # run the main routine of QUEENS
    arguments = [
        "--input=" + input_file,
        "--output=" + str(tmpdir),
    ]

    # This seed is fixed so that the variational distribution is initialized so that the park
    # function can be evaluated correctly
    np.random.seed(211)
    # actual main call of vi_rp
    main(arguments)

    # get the results of the QUEENS run
    result_file = os.path.join(tmpdir, "inverse_rpvi_park91a_hifi.pickle")
    with open(result_file, "rb") as handle:
        results = pickle.load(handle)

    # Actual tests
    assert np.abs(results["variational_distribution"]["mean"][0] - 0.5) < 0.25
    assert np.abs(results["variational_distribution"]["mean"][1] - 0.2) < 0.1
    assert results["variational_distribution"]["covariance"][0, 0] ** 0.5 < 0.5
    assert results["variational_distribution"]["covariance"][1, 1] ** 0.5 < 0.5


@pytest.fixture()
def design_and_write_experimental_data_to_csv(tmpdir):
    """Generate artificial experimental data."""
    # Fix random seed
    np.random.seed(seed=1)

    # create target inputs
    x1 = 0.5
    x2 = 0.2

    # use x3 and x4 as coordinates and create coordinate grid (same as in park91a_hifi_coords)
    xx3 = np.linspace(0, 1, 4)
    xx4 = np.linspace(0, 1, 4)
    x3_vec, x4_vec = np.meshgrid(xx3, xx4)
    x3_vec = x3_vec.flatten()
    x4_vec = x4_vec.flatten()

    # generate clean function output for fake test data
    y_vec = []
    for x3, x4 in zip(x3_vec, x4_vec):
        y_vec.append(park91a_hifi_coords(x1, x2, x3, x4))
    y_vec = np.array(y_vec)

    # add artificial noise to fake measurements
    sigma_n = 0.001
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
