"""Integration test for the classification iterator."""
import os
import pickle
from pathlib import Path

import numpy as np

from pqueens import run
from pqueens.utils.injector import inject


def test_classification_iterator(inputdir, tmpdir):
    """Integration test for the classfication iterator."""
    input_template = Path(inputdir).joinpath('classification_nn_rosenbrock_template.yaml')
    plotting_dir = str(tmpdir)
    classification_funcation_path = Path(tmpdir).joinpath("classification_function.py")

    # create classification function
    with classification_funcation_path.open("w", encoding="utf-8") as f:
        f.write("classify = lambda x: x>80")

    inject(
        {"plotting_dir": plotting_dir, "external_module_path": str(classification_funcation_path)},
        str(input_template),
        input_path := str(tmpdir.join("classification_nn_rosenbrock.yaml")),
    )
    run(Path(input_path), Path(tmpdir))
    result_file = os.path.join(tmpdir, 'classification_nn_rosenbrock.pickle')
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    expected_results = np.ones((12, 1))
    expected_results[-2:] = 0
    expected_samples = np.array(
        [
            [-1.08480734, -0.7215275],
            [0.47755988, -1.70552695],
            [1.2057831, -0.18745467],
            [-0.82740908, -1.73532511],
            [-0.60210352, 1.98595546],
            [-0.66345435, 1.99530424],
            [-0.46880494, 1.98952451],
            [-0.31552438, 1.98624568],
            [-0.86489638, -0.2715313],
            [1.87330784, 1.77265221],
            [0.88234568, 0.14938235],
            [0.3827193, 0.81934479],
        ]
    )
    obtained_results = results["classified_outputs"]
    obtained_samples = results["input_samples"]
    np.testing.assert_array_almost_equal(expected_samples, obtained_samples)
    np.testing.assert_array_equal(expected_results, obtained_results)
