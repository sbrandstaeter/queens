"""Numpy array utils."""
import numpy as np


def at_least_2d(arr):
    """View input array as array with at least two dimensions.

    Args:
        arr (array_like): Input array

    Returns:
        arr (np.ndarray): View of input array with at least two dimensions
    """
    arr = np.array(arr)
    if arr.ndim == 0:
        return arr.reshape((1, 1))
    elif arr.ndim == 1:
        return arr[:, np.newaxis]
    else:
        return arr


def at_least_3d(arr):
    """View input array as array with at least three dimensions.

    Args:
        arr (array_like): Input array

    Returns:
        arr (np.ndarray): View of input array with at least three dimensions
    """
    arr = np.array(arr)
    if arr.ndim == 0:
        return arr.reshape((1, 1, 1))
    elif arr.ndim == 1:
        return arr[:, np.newaxis, np.newaxis]
    elif arr.ndim == 2:
        return arr[:, :, np.newaxis]
    else:
        return arr
