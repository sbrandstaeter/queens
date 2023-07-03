"""Test suite for integration tests for the Morris-Salib Iterator.

Estimate Elementary Effects for local simulations with BACI using the
INVAAA minimal model.
"""

from pathlib import Path

from pqueens import run
from pqueens.utils import injector


def test_baci_elementary_effects(
    tmp_path,
    inputdir,
    third_party_inputs,
    baci_link_paths,
    baci_elementary_effects_check_results,
):
    """Integration test for the Elementary Effects Iterator together with BACI.

    The test runs a local native BACI simulation as well as a local
    Singularity based BACI simulation for elementary effects.
    """
    template = inputdir / "baci_local_elementary_effects_template_dask.yml"
    input_file = tmp_path / "elementary_effects_baci_local_invaaa.yml"
    third_party_input_file = third_party_inputs / "baci_input_files/invaaa_ee.dat"
    experiment_name = "ee_invaaa_local"

    baci_release, post_drt_monitor, _, _ = baci_link_paths

    dir_dict = {
        'experiment_name': experiment_name,
        'baci_input': third_party_input_file,
        'baci_release': baci_release,
        'post_drt_monitor': post_drt_monitor,
    }

    injector.inject(dir_dict, template, input_file)
    run(Path(input_file), Path(tmp_path))

    result_file_name = experiment_name + ".pickle"
    result_file = tmp_path / result_file_name
    baci_elementary_effects_check_results(result_file)
