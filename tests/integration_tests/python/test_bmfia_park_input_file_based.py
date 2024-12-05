#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (c) 2024, QUEENS contributors.
#
# This file is part of QUEENS.
#
# QUEENS is free software: you can redistribute it and/or modify it under the terms of the GNU
# Lesser General Public License as published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version. QUEENS is distributed in the hope that it will
# be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details. You
# should have received a copy of the GNU Lesser General Public License along with QUEENS. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Integration tests for the BMFIA using an input file."""

# pylint: disable=invalid-name
from pathlib import Path

import numpy as np
import pytest

from queens.main import run
from queens.utils import injector
from queens.utils.io_utils import load_result


@pytest.fixture(name="expected_variational_mean")
def fixture_expected_variational_mean():
    """Expected variational mean values."""
    exp_var_mean = np.array([0.53, 0.53]).reshape(-1, 1)

    return exp_var_mean


@pytest.mark.max_time_for_test(30)
def test_bmfia_smc_park(
    inputdir,
    tmp_path,
    _create_experimental_data_park91a_hifi_on_grid,
    expected_samples,
    expected_weights,
):
    """Integration test for BMFIA.

    Integration test for bayesian multi-fidelity inverse analysis
    (bmfia) using the park91 function.
    """
    # generate yml input file from template
    template = inputdir / "bmfia_smc_park.yml"
    experimental_data_path = tmp_path
    dir_dict = {"experimental_data_path": experimental_data_path, "plot_dir": tmp_path}
    input_file = tmp_path / "smc_mf_park_realization.yml"
    injector.inject(dir_dict, template, input_file)

    # run the main routine of QUEENS
    run(Path(input_file), Path(tmp_path))

    # get the results of the QUEENS run
    result_file = tmp_path / "smc_park_mf.pickle"
    results = load_result(result_file)

    samples = results["raw_output_data"]["particles"].squeeze()
    weights = results["raw_output_data"]["weights"].squeeze()

    # some tests / asserts here
    np.testing.assert_array_almost_equal(samples, expected_samples, decimal=5)
    np.testing.assert_array_almost_equal(weights.flatten(), expected_weights.flatten(), decimal=5)


@pytest.mark.max_time_for_test(20)
def test_bmfia_rpvi_gp_park(
    inputdir,
    tmp_path,
    _create_experimental_data_park91a_hifi_on_grid,
    expected_variational_mean,
    expected_variational_cov,
):
    """Integration test for BMFIA.

    Integration test for bayesian multi-fidelity inverse analysis
    (bmfia) using the park91 function.
    """
    template = inputdir / "bmfia_rpvi_park_gp_template.yml"
    experimental_data_path = tmp_path
    dir_dict = {
        "experimental_data_path": experimental_data_path,
        "plot_dir": tmp_path,
    }
    input_file = tmp_path / "bmfia_rpvi_park.yml"
    injector.inject(dir_dict, template, input_file)

    # run the main routine of QUEENS
    run(Path(input_file), Path(tmp_path))

    # get the results of the QUEENS run
    result_file = tmp_path / "bmfia_rpvi_park.pickle"
    results = load_result(result_file)

    variational_mean = results["variational_distribution"]["mean"]
    variational_cov = results["variational_distribution"]["covariance"]

    # some tests / asserts here
    np.testing.assert_array_almost_equal(variational_mean, expected_variational_mean, decimal=2)
    np.testing.assert_array_almost_equal(variational_cov, expected_variational_cov, decimal=2)


def test_bmfia_rpvi_NN_park(
    inputdir,
    tmp_path,
    _create_experimental_data_park91a_hifi_on_grid,
    expected_variational_mean_nn,
    expected_variational_cov_nn,
):
    """Integration test for BMFIA.

    Integration test for bayesian multi-fidelity inverse analysis
    (bmfia) using the park91 function.
    """
    template = inputdir / "bmfia_rpvi_park_NN_template.yml"
    experimental_data_path = tmp_path
    dir_dict = {
        "experimental_data_path": experimental_data_path,
        "plot_dir": tmp_path,
    }
    input_file = tmp_path / "bmfia_rpvi_park.yml"
    injector.inject(dir_dict, template, input_file)

    # run the main routine of QUEENS
    run(Path(input_file), Path(tmp_path))

    # get the results of the QUEENS run
    result_file = tmp_path / "bmfia_rpvi_park.pickle"
    results = load_result(result_file)

    variational_mean = results["variational_distribution"]["mean"]
    variational_cov = results["variational_distribution"]["covariance"]

    # some tests / asserts here
    np.testing.assert_array_almost_equal(variational_mean, expected_variational_mean_nn, decimal=1)
    np.testing.assert_array_almost_equal(variational_cov, expected_variational_cov_nn, decimal=1)
