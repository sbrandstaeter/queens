"""Test-module for numpy utils functions."""
import numpy as np
import pytest

from pqueens.utils.numpy_utils import at_least_2d, at_least_3d


@pytest.fixture(name="arr_0d", scope='module')
def arr_0d_fixture():
    """Return possible number of weights."""
    return np.random.rand(1).squeeze()


@pytest.fixture(name="arr_1d", scope='module')
def arr_1d_fixture():
    """Return 1D array."""
    return np.random.rand(3)


@pytest.fixture(name="arr_2d", scope='module')
def arr_2d_fixture():
    """Return 2D array."""
    return np.random.rand(4, 2)


@pytest.fixture(name="arr_3d", scope='module')
def arr_3d_fixture():
    """Return 3D array."""
    return np.random.rand(3, 2, 5)


def test_at_least_2d(arr_0d, arr_1d, arr_2d):
    """Test numpy utils function *at_least_2d*."""
    np.testing.assert_equal(at_least_2d(arr_0d).shape, (1, 1))
    np.testing.assert_equal(at_least_2d(arr_1d).shape, (arr_1d.shape[0], 1))
    np.testing.assert_equal(at_least_2d(arr_2d).shape, arr_2d.shape)


def test_at_least_3d(arr_0d, arr_1d, arr_2d, arr_3d):
    """Test numpy utils function *at_least_3d*."""
    np.testing.assert_equal(at_least_3d(arr_0d).shape, (1, 1, 1))
    np.testing.assert_equal(at_least_3d(arr_1d).shape, (arr_1d.shape[0], 1, 1))
    np.testing.assert_equal(at_least_3d(arr_2d).shape, (arr_2d.shape[0], arr_2d.shape[1], 1))
    np.testing.assert_equal(at_least_3d(arr_3d).shape, arr_3d.shape)
