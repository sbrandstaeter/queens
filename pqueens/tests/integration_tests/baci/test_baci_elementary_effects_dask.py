"""Test suite for integration tests for the Morris-Salib Iterator.

Estimate Elementary Effects for local simulations with BACI using the
INVAAA minimal model.
"""

import json
import os
from pathlib import Path

from pqueens import run
from pqueens.utils import injector


def test_baci_elementary_effects(
    tmpdir,
    inputdir,
    third_party_inputs,
    baci_link_paths,
    singularity_bool,
    baci_elementary_effects_check_results,
):
    """Integration test for the Elementary Effects Iterator together with BACI.

    The test runs a local native BACI simulation as well as a local Singularity
    based BACI simulation for elementary effects.

    Args:
        tmpdir (Path):
        inputdir (str): Path to the JSON input file
        third_party_inputs (str): Path to the BACI input files
        baci_link_paths(str): Path to the links pointing to *baci_release* and *post_drt_monitor*
        singularity_bool (str): String that encodes a boolean that is parsed to the JSON input file
        baci_elementary_effects_check_results (function): function to check the results
    """
    template = os.path.join(inputdir, "baci_local_elementary_effects_template_dask.yml")
    input_file = os.path.join(tmpdir, "elementary_effects_baci_local_invaaa.yml")
    third_party_input_file = os.path.join(third_party_inputs, "baci_input_files", "invaaa_ee.dat")
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
    run(Path(input_file), Path(tmpdir))

    result_file_name = experiment_name + ".pickle"
    result_file = os.path.join(tmpdir, result_file_name)
    baci_elementary_effects_check_results(result_file)
