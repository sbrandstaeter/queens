"""Integration test for the classification iterator."""

import numpy as np

from queens.main import run
from queens.utils.injector import inject
from queens.utils.pickle_utils import load_pickle


def test_classification_iterator(inputdir, tmp_path):
    """Integration test for the classification iterator."""
    input_template = inputdir / "classification_nn_rosenbrock_template.yaml"
    classification_function_path = tmp_path / "classification_function.py"

    # create classification function called classify
    classification_function_name = "classify"
    classification_function_path.write_text(
        f"{classification_function_name} = lambda x: x>80", encoding="utf-8"
    )
    experiment_name = "classification_nn_rosenbrock"
    input_path = tmp_path / f"{experiment_name}.yaml"
    inject(
        params={
            "experiment_name": experiment_name,
            "plotting_dir": str(tmp_path),
            "classification_function_name": classification_function_name,
            "external_module_path": str(classification_function_path),
        },
        template_path=input_template,
        output_file=input_path,
    )
    run(input_path, tmp_path)
    result_file = tmp_path / f"{experiment_name}.pickle"
    results = load_pickle(result_file)

    expected_results_classified = np.ones((12, 1))
    expected_results_classified[-2:] = 0
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
    np.testing.assert_array_equal(expected_results_classified, obtained_results)
