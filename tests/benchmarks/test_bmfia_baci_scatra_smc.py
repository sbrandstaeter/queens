"""TODO_doc."""

import pytest

from queens.main import run
from queens.utils import injector
from queens.utils.io_utils import load_result
from test_utils.benchmarks import assert_weights_and_samples


def test_bmfia_baci_scatra_smc(inputdir, tmp_path, dir_dict, expected_weights, expected_samples):
    """TODO_doc: add a one-line description.

    Integration test for smc with a simple diffusion problem (scatra) in
    BACI.
    """
    # generate yaml input file from template
    template = inputdir / 'bmfia_scatra_baci_template_smc_gp_precompiled_copy.yml'
    input_file = tmp_path / 'hf_scatra_baci.yml'
    injector.inject(dir_dict, template, input_file)

    # run the main routine of QUEENS
    run(input_file, tmp_path)

    # get the results of the QUEENS run
    results = load_result(tmp_path / "bmfia_baci_scatra_smc.pickle")

    assert_weights_and_samples(results, expected_weights, expected_samples)


@pytest.fixture(name="expected_weights")
def fixture_expected_weights():
    """TODO_doc."""
    weights = 1
    return weights


@pytest.fixture(name="expected_samples")
def fixture_expected_samples():
    """TODO_doc."""
    samples = 1
    return samples
