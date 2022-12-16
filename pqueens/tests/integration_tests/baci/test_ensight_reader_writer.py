import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from pqueens import run
from pqueens.utils import injector


def test_ensight_reader_writer(
    inputdir, tmpdir, third_party_inputs, baci_link_paths, config_dir, expected_mean, expected_var
):
    # generate json input file from template
    third_party_input_file = os.path.join(third_party_inputs, "baci_input_files", "invaaa_ee.dat")
    baci_release, _, post_drt_ensight, _ = baci_link_paths
    dir_dict = {
        'baci_input': third_party_input_file,
        'post_drt_ensight': post_drt_ensight,
        'baci-release': baci_release,
    }
    template = os.path.join(inputdir, "baci_ensight_template.yml")
    input_file = os.path.join(tmpdir, "baci_ensight.yml")
    injector.inject(dir_dict, template, input_file)

    # get json file as config dictionary
    run(Path(input_file), Path(tmpdir))

    # run a MC simulation with random input for now

    # Check if we got the expected results
    experiment_name = "baci_ensight"
    result_file_name = experiment_name + ".pickle"

    result_file = os.path.join(str(tmpdir), result_file_name)
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    # assert statements
    np.testing.assert_array_almost_equal(results['mean'], expected_mean, decimal=4)
    np.testing.assert_array_almost_equal(results['var'], expected_var, decimal=4)


@pytest.fixture()
def expected_mean():
    result = np.array(
        [
            [1.74423399, 4.33662133],
            [1.74423399, 4.33662133],
            [2.04178376, 3.26816187],
            [2.04178376, 3.26816187],
            [1.82349517, 2.04028533],
            [1.82349517, 2.04028533],
            [1.05847878, 0.83544016],
            [1.05847878, 0.83544016],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.17496549, 5.37049122],
            [0.17496549, 5.37049122],
            [1.06617833, 5.07653875],
            [1.06617833, 5.07653875],
        ]
    )
    return result


@pytest.fixture()
def expected_var():
    result = np.array(
        [
            [0.03219374, 0.23187617],
            [0.03219374, 0.23187617],
            [0.04138086, 0.14548341],
            [0.04138086, 0.14548341],
            [0.02985786, 0.07354971],
            [0.02985786, 0.07354971],
            [0.00941438, 0.02664331],
            [0.00941438, 0.02664331],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.00165357, 0.34432005],
            [0.00165357, 0.34432005],
            [0.01429646, 0.30811321],
            [0.01429646, 0.30811321],
        ]
    )
    return result
