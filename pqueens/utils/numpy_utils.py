"""Numpy array utils."""
import logging

import numpy as np

_logger = logging.getLogger(__name__)


def at_least_2d(arr):
    """View input array as array with at least two dimensions.

    Args:
        arr (np.ndarray): Input array

    Returns:
        arr (np.ndarray): View of input array with at least two dimensions
    """
    if arr.ndim == 0:
        return arr.reshape((1, 1))
    elif arr.ndim == 1:
        return arr[:, np.newaxis]
    else:
        return arr


def at_least_3d(arr):
    """View input array as array with at least three dimensions.

    Args:
        arr (np.ndarray): Input array

    Returns:
        arr (np.ndarray): View of input array with at least three dimensions
    """
    if arr.ndim == 0:
        return arr.reshape((1, 1, 1))
    elif arr.ndim == 1:
        return arr[:, np.newaxis, np.newaxis]
    elif arr.ndim == 2:
        return arr[:, :, np.newaxis]
    else:
        return arr


def safe_cholesky(matrix):
    """Numerically stable Cholesky decomposition.

    Compute the Cholesky decomposition of a matrix. Numeric stability is increased by
    sequentially adding a small term to the diagonal of the matrix.

    Args:
        matrix (np.ndarray): Matrix to be decomposed

    Returns:
        low_cholesky (np.ndarray): lower-triangular Cholesky factor of matrix
    """
    try:
        low_cholesky = np.linalg.cholesky(matrix)
        return low_cholesky
    except np.linalg.LinAlgError:
        matrix_max = np.max(matrix)
        for i in range(5):
            jitter = matrix_max * 1e-10 * 10**i
            matrix_ = matrix + np.eye(matrix.shape[0]) * jitter
            _logger.warning(
                'Added {:.2e} to diagonal of matrix for numerical stability '
                'of cholesky decompostition'.format(jitter)
            )
            try:
                low_cholesky = np.linalg.cholesky(matrix_)
                return low_cholesky
            except np.linalg.LinAlgError:
                continue
        raise np.linalg.LinAlgError('Cholesky decomposition failed due to ill-conditioning!')
