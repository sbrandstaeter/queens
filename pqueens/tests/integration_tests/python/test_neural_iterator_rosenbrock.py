import os
import pickle

import numpy as np
import pytest

from pqueens.main import main


@pytest.mark.integration_tests
def test_neural_iterator(inputdir, tmpdir):
    """Integration test for the neural iterator."""
    arguments = [
        '--input=' + os.path.join(inputdir, 'neural_iterator_rosenbrock.json'),
        '--output=' + str(tmpdir),
    ]

    main(arguments)
    result_file = os.path.join(tmpdir, 'neural_iterator_rosenbrock.pickle')
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    np.testing.assert_array_equal(
        results["raw_output_data"]["mean"].shape,
        (300, 1),
    )

    np.testing.assert_array_equal(
        results["input_data"].shape,
        (300, 2),
    )
