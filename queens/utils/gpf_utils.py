"""Utilis for gpflow."""

from typing import TYPE_CHECKING

import numpy as np
from sklearn.preprocessing import StandardScaler

# This allows autocomplete in the IDE
if TYPE_CHECKING:
    import gpflow as gpf
else:
    from queens.utils.import_utils import LazyLoader

    gpf = LazyLoader("gpflow")


def init_scaler(unscaled_data):
    r"""Initialize StandardScaler and scale data.

    Standardize features by removing the mean and scaling to unit variance

        :math:`scaled\_data = \frac{unscaled\_data - mean}{std}`

    Args:
        unscaled_data (np.ndarray): Unscaled data

    Returns:
        scaler (StandardScaler): Standard scaler
        scaled_data (np.ndarray): Scaled data
    """
    scaler = StandardScaler()
    scaler.fit(unscaled_data)
    scaled_data = scaler.transform(unscaled_data)
    return scaler, scaled_data


def set_transform_function(data, transform):
    """Set transform function.

    Args:
        data (gpf.Parameter): Data to be transformed
        transform (tfp.bijectors.Bijector): Transform function

    Returns:
        gpf.Parameter with transform
    """
    return gpf.Parameter(
        data,
        name=data.name.split(":")[0],
        transform=transform,
    )


def extract_block_diag(array, block_size):
    """Extract block diagonals of square 2D Array.

    Args:
        array (np.ndarray): Square 2D array
        block_size (int): Block size

    Returns:
        3D Array containing block diagonals
    """
    n_blocks = array.shape[0] // block_size

    new_shape = (n_blocks, block_size, block_size)
    new_strides = (
        block_size * array.strides[0] + block_size * array.strides[1],
        array.strides[0],
        array.strides[1],
    )

    return np.lib.stride_tricks.as_strided(array, new_shape, new_strides)
