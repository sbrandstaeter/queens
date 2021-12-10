import os
import pickle

import numpy as np
import pytest

from pqueens.main import main


@pytest.mark.integration_tests
def test_grid_iterator(inputdir, tmpdir, expected_response, expected_grid):
    """Integration test for the grid iterator."""
    arguments = [
        '--input=' + os.path.join(inputdir, 'grid_iterator_local.json'),
        '--output=' + str(tmpdir),
    ]

    main(arguments)
    result_file = os.path.join(tmpdir, 'grid_iterator_local.pickle')
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    np.testing.assert_array_equal(
        results["raw_output_data"]["mean"], expected_response,
    )

    np.testing.assert_allclose(results["input_data"], expected_grid, rtol=1.0e-3)


@pytest.fixture()
def expected_grid():
    input_data = np.array(
        [
            [-2.000, -2.000],
            [-1.000, -2.000],
            [0.000, -2.000],
            [1.000, -2.000],
            [2.000, -2.000],
            [-2.000, -1.000],
            [-1.000, -1.000],
            [0.000, -1.000],
            [1.000, -1.000],
            [2.000, -1.000],
            [-2.000, 0.000],
            [-1.000, 0.000],
            [0.000, 0.000],
            [1.000, 0.000],
            [2.000, 0.000],
            [-2.000, 1.000],
            [-1.000, 1.000],
            [0.000, 1.000],
            [1.000, 1.000],
            [2.000, 1.000],
            [-2.000, 2.000],
            [-1.000, 2.000],
            [0.000, 2.000],
            [1.000, 2.000],
            [2.000, 2.000],
        ]
    )
    return input_data


@pytest.fixture()
def expected_response():
    expected_response = np.atleast_2d(
        np.array(
            [
                3.609e03,
                9.040e02,
                4.010e02,
                9.000e02,
                3.601e03,
                2.509e03,
                4.040e02,
                1.010e02,
                4.000e02,
                2.501e03,
                1.609e03,
                1.040e02,
                1.000e00,
                1.000e02,
                1.601e03,
                9.090e02,
                4.000e00,
                1.010e02,
                0.000e00,
                9.010e02,
                4.090e02,
                1.040e02,
                4.010e02,
                1.000e02,
                4.010e02,
            ]
        )
    ).T

    return expected_response
