"""TODO_doc."""

import pytest

from queens.main import run
from queens.utils import injector
from queens.utils.io_utils import load_result
from test_utils.benchmarks import assert_weights_and_samples


def test_bmfia_fourc_scatra_smc(
    inputdir, tmp_path, paths_dictionary, expected_weights, expected_samples
):
    """TODO_doc: add a one-line description.

    Integration test for smc with a simple diffusion problem (scatra) in
    fourc.
    """
    # generate yaml input file from template
    template = inputdir / "bmfia_scatra_fourc_template_smc_gp_precompiled_copy.yml"
    input_file = tmp_path / "hf_scatra_fourc.yml"
    injector.inject(paths_dictionary, template, input_file)

    # run the main routine of QUEENS
    run(input_file, tmp_path)

    # Load results
    result_file = tmp_path / "dummy_experiment_name.pickle"
    results = load_result(result_file)

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
