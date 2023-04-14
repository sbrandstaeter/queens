"""Test suite for integration tests for the Morris-Salib Iterator.

Estimate Elementary Effects for local simulations with BACI using the
INVAAA minimal model.
"""

import json
import shutil

import pytest

from pqueens import run
from pqueens.utils import injector


@pytest.fixture(scope="session")
def output_directory_forward(tmp_path_factory):
    """Create two temporary output directories for test runs with singularity.

        * with singularity (<...>_true)
        * without singularity (<...>_false)

    Args:
        tmp_path_factory: Fixture used to create arbitrary temporary directories

    Returns:
        output_directory_forward (dict): Temporary output directories for simulation without and
        with singularity
    """
    path_singularity_true = tmp_path_factory.mktemp("test_baci_elementary_effects_true")
    path_singularity_false = tmp_path_factory.mktemp("test_baci_elementary_effects_false")

    return {True: path_singularity_true, False: path_singularity_false}


@pytest.fixture()
def experiment_directory(output_directory_forward, singularity_bool):
    """Return experiment directory depending on *singularity_bool*.

    Returns:
        experiment_directory (LocalPath): Experiment directory depending on *singularity_bool*
    """
    return output_directory_forward[singularity_bool]


@pytest.fixture()
def check_experiment_directory(experiment_directory):
    """Check if experiment directory contains subdirectories.

    Raises:
        AssertionError: If experiment directory does not contain subdirectories.
    """
    number_subdirectories = count_subdirectories(experiment_directory)

    assert (
        number_subdirectories != 0
    ), "Empty output directory. Run test_baci_elementary_effects first."


def count_subdirectories(current_directory):
    """Count subdirectories in *current_directory*.

    Returns:
        number_subdirectories (int): Number of subdirectories
    """
    number_subdirectories = 0
    for current_subdirectory in current_directory.iterdir():
        if current_subdirectory.is_dir():
            number_subdirectories += 1
    return number_subdirectories


def remove_job_output_directory(experiment_directory, jobid):
    """Remove output directory of job #jobid from *experiment_directory*."""
    shutil.rmtree(experiment_directory / str(jobid))


def test_baci_elementary_effects(
    inputdir,
    third_party_inputs,
    baci_link_paths,
    singularity_bool,
    experiment_directory,
    baci_elementary_effects_check_results,
):
    """Integration test for the Elementary Effects Iterator together with BACI.

    The test runs a local native BACI simulation as well as a local Singularity
    based BACI simulation for elementary effects.

    Args:
        inputdir (Path): Path to the JSON input file
        third_party_inputs (Path): Path to the BACI input files
        baci_link_paths(Path): Path to the links pointing to *baci-release* and *post_drt_monitor*
        singularity_bool (str): String that encodes a boolean that is parsed to the JSON input file
        experiment_directory (LocalPath): Experiment directory depending on *singularity_bool*
        baci_elementary_effects_check_results (function): function to check the results
    """
    template = inputdir / "baci_local_elementary_effects_template.yml"
    input_file = experiment_directory / "elementary_effects_baci_local_invaaa.yml"
    third_party_input_file = third_party_inputs / "baci_input_files" / "invaaa_ee.dat"
    experiment_name = "ee_invaaa_local_singularity_" + json.dumps(singularity_bool)

    baci_release, post_drt_monitor, _, _ = baci_link_paths

    dir_dict = {
        'experiment_name': experiment_name,
        'baci_input': third_party_input_file,
        'baci_release': baci_release,
        'post_drt_monitor': post_drt_monitor,
        'singularity_boolean': json.dumps(singularity_bool),
    }

    injector.inject(dir_dict, template, input_file)
    run(input_file, experiment_directory)

    result_file = experiment_directory / (experiment_name + ".pickle")

    # test results of SA analysis
    baci_elementary_effects_check_results(result_file)
