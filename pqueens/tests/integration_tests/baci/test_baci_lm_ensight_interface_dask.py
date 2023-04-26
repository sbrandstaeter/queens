"""Test suite for the heritage BACI Levenberg-Marquardt optimizer.

Test local simulations with BACI using a minimal FSI model and the
*data_processor_ensight_interface* evaluation and therefore
*post_drt_ensight* post-processor.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pqueens import run
from pqueens.utils import injector


def test_baci_lm_shape(
    tmpdir,
    inputdir,
    third_party_inputs,
    baci_link_paths,
):
    """Integration test for the Baci Levenberg Marquardt Iterator with BACI.

    The test runs local native BACI simulations as well as a local
    Singularity based BACI simulations.

    Args:
        inputdir (str): Path to the JSON input file
        third_party_inputs (str): Path to the BACI input files
        baci_link_paths (str): Symbolic links to executables including BACI
    """
    template = os.path.join(inputdir, "baci_local_shape_lm_template_dask.yml")
    input_file = os.path.join(tmpdir, "baci_local_shape_lm.yml")
    third_party_input_file = os.path.join(
        third_party_inputs, "baci_input_files", "lm_tri_fsi_shape_template.dat"
    )
    third_party_input_file_monitor = os.path.join(
        third_party_inputs, "baci_input_files", "lm_tri_fsi_shape_E2000_nue03_p.monitor"
    )
    experiment_name = "OptmizeBaciLM"

    baci_release, _, _, post_processor = baci_link_paths

    dir_dict = {
        'experiment_name': experiment_name,
        'baci_input': third_party_input_file,
        'baci_input_monitor': third_party_input_file_monitor,
        'baci_release': baci_release,
        'post_processor': post_processor,
    }

    injector.inject(dir_dict, template, input_file)
    run(Path(input_file), Path(tmpdir))

    result_file_name = experiment_name + ".csv"
    result_file = os.path.join(tmpdir, result_file_name)

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
    sys.exit(0)  # Trigger dask client shutdown
