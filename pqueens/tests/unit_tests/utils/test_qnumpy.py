"""Test data class."""
import numpy as np
import pytest

import pqueens.utils.qnumpy as qnp


@pytest.fixture(scope='module')
def arrays():
    """Create QArrays."""
    np.random.seed(42)
    np_arr_1 = np.random.rand(10, 3, 8)
    np_arr_2 = np.random.rand(3, 8, 10)

    arr1 = qnp.array(np_arr_1, ['samples', 'out_dim', 'coordinates'])
    arr2 = qnp.array(np_arr_2, ['out_dim', 'coordinates', 'samples'])

    return arr1, arr2


@pytest.mark.unit_tests
def test_dot_product(arrays, expected_result):
    """Test the dot product."""
    res = qnp.dot_product(arrays[0], arrays[1], ['samples', 'coordinates'])
    np_res = res.to_numpy(['out_dim_2', 'out_dim_1'])
    np.testing.assert_array_almost_equal(np_res, expected_result, decimal=8)


@pytest.fixture()
def expected_result():
    """Expected result of dot product."""
    result = np.array(
        [
            [20.94645047, 20.16812015, 21.40561346],
            [19.09014715, 19.64010639, 19.67489996],
            [18.65635214, 18.78947260, 20.37398357],
        ]
    )
    return result
