"""Test suite for the heritage BACI Levenberg-Marquardt optimizer.

Test local simulations with BACI using a minimal FSI model and the
*data_processor_ensight_interface* evaluation and therefore
*post_drt_ensight* post-processor.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
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
    path_singularity_true = tmp_path_factory.mktemp("test_baci_lm_shape_singularity")
    path_singularity_false = tmp_path_factory.mktemp("test_baci_lm_shape_nosingularity")

    return {True: path_singularity_true, False: path_singularity_false}


@pytest.fixture()
def experiment_directory(output_directory_forward, singularity_bool):
    """Return experiment directory depending on *singularity_bool*.

    Returns:
        experiment_directory (LocalPath): Experiment directory depending on *singularity_bool*
    """
    return output_directory_forward[singularity_bool]


def test_baci_lm_shape(
    inputdir,
    third_party_inputs,
    baci_link_paths,
    singularity_bool,
    experiment_directory,
):
    """Integration test for the Baci Levenberg Marquardt Iterator with BACI.

    The test runs local native BACI simulations as well as a local
    Singularity based BACI simulations.

    Args:
        inputdir (str): Path to the JSON input file
        third_party_inputs (str): Path to the BACI input files
        baci_link_paths (str): Symbolic links to executables including BACI
        singularity_bool (str): String that encodes a boolean that is parsed to the JSON input file
        experiment_directory (LocalPath): Experiment directory depending on *singularity_bool*
    """
    template = Path(inputdir, "baci_local_shape_lm_template.yml")
    input_file = Path(experiment_directory, "baci_local_shape_lm.yml")
    third_party_input_file = Path(
        third_party_inputs, "baci_input_files", "lm_tri_fsi_shape_template.dat"
    )
    third_party_input_file_monitor = Path(
        third_party_inputs, "baci_input_files", "lm_tri_fsi_shape_E2000_nue03_p.monitor"
    )
    experiment_name = "OptmizeBaciLM_" + json.dumps(singularity_bool)

    baci_release, _, _, post_processor = baci_link_paths

    dir_dict = {
        'experiment_name': experiment_name,
        'baci_input': third_party_input_file,
        'baci_input_monitor': third_party_input_file_monitor,
        'baci-release': baci_release,
        'post_processor': post_processor,
        'singularity_boolean': json.dumps(singularity_bool),
    }

    injector.inject(dir_dict, template, input_file)
    run(input_file, experiment_directory)

    result_file_name = experiment_name + ".csv"
    result_file = Path(experiment_directory, result_file_name)

    result_data = pd.read_csv(
        result_file,
        sep='\t',
    )

    np.testing.assert_equal(result_data['iter'][1], 1)
    np.testing.assert_allclose(result_data['resnorm'][1], 1.42069484e-02, 1.0e-5)
    np.testing.assert_allclose(result_data['gradnorm'][1], 4.53755307e-03, 1.0e-5)

    params = result_data["params"].str.replace(r"[", "", regex=True)
    params = params.str.replace(r"]", "", regex=True)
    params = params.str.split(expand=True)
    np.testing.assert_allclose(params.loc[1].astype(float), [1.70123614e03, 3.43936558e-01], 1.0e-5)

    delta_params = result_data["delta_params"].str.replace(r"[", "", regex=True)
    delta_params = delta_params.str.replace(r"]", "", regex=True)
    delta_params = delta_params.str.split(expand=True)
    np.testing.assert_allclose(
        delta_params.loc[1].astype(float), [1.28123570e02, -3.47241922e-02], 1.0e-5
    )

    np.testing.assert_allclose(result_data['mu'][1], 1.0, 1.0e-5)
