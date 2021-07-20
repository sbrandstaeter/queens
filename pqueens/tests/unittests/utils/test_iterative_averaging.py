from pqueens.utils.iterative_averaging_utils import *
import numpy as np


def test_L1_norm():
    x = 2 * np.ones(10)
    norm_L1 = L1_norm(x)
    norm_L1_avg = L1_norm(x, True)
    np.testing.assert_almost_equal(norm_L1, 20)
    np.testing.assert_almost_equal(norm_L1_avg, 2)


def test_L2_norm():
    x = 2 * np.ones(10)
    norm_L2 = L2_norm(x)
    norm_L2_avg = L2_norm(x, True)
    np.testing.assert_almost_equal(norm_L2, 2 * np.sqrt(10))
    np.testing.assert_almost_equal(norm_L2_avg, 2)


def test_relative_change():
    old = np.ones(10)
    new = np.ones(10) * 2
    rel_change = relative_change(old, new, L1_norm)
    np.testing.assert_almost_equal(rel_change, 1)


def test_polyak_averaging():
    polyak = PolyakAveraging(None, 0, 1, 1, 1, 0)
    for j in range(10):
        polyak.update_average(j)
    np.testing.assert_almost_equal(polyak.current_average, np.mean(np.arange(10)))


def test_moving_averaging():
    moving = MovingAveraging(None, 0, 1, 1, 5, [])
    for j in range(10):
        moving.update_average(np.ones(5) * j)
    np.testing.assert_almost_equal(moving.current_average, np.mean(np.arange(0, 10)[-5:]))


def test_exponential_averaging():
    alpha = 0.25
    exponential_avg = ExponentialAveraging(None, 0, 1, 1, alpha)
    for j in range(10):
        exponential_avg.update_average(j)
    # For this special case there is a analytical solution
    ref = np.sum((1 - alpha) * np.arange(1, 10) * alpha ** np.arange(0, 9)[::-1])
    np.testing.assert_almost_equal(exponential_avg.current_average, ref)

