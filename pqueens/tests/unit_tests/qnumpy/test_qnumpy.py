"""Test data class."""
import numpy as np
import pytest

import pqueens.qnumpy.qnumpy as qnp


@pytest.fixture(scope='module')
def arrays():
    """Create QArrays."""
    np_arr_1 = np.random.rand(10, 3, 8)
    np_arr_2 = np.random.rand(3, 8, 10)

    arr1 = qnp.QArray(np_arr_1, ['samples', 'out_dim', 'coordinates'])
    arr2 = qnp.QArray(np_arr_2, ['out_dim', 'coordinates', 'samples'])

    return arr1, arr2


@pytest.mark.unit_tests
def test_dot_product(arrays):
    """Test the dot product."""
    res = qnp.dot_product(arrays[0], arrays[1], ['samples', 'coordinates'])
