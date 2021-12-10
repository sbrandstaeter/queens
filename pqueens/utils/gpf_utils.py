import gpflow as gpf
import numpy as np
from sklearn.preprocessing import StandardScaler


def init_scaler(unscaled_data):
    """Initialize StandardScaler and scale data.

    Standardize features by removing the mean and scaling to unit variance

        scaled_data = (unscaled_data - mean) / std

    Args:
        unscaled_data (np.ndarray): unscaled data

    Returns:
        scaler (StandardScaler): standard scaler
        scaled_data (np.ndarray): scaled data
    """
    scaler = StandardScaler()
    scaler.fit(unscaled_data)
    scaled_data = scaler.transform(unscaled_data)
    return scaler, scaled_data


def set_transform_function(data, transform):
    """Set transform function.

    Args:
        data (gpf.Parameter): data to be transformed
        transform (tfp.bijectors.Bijector): transform function

    Returns:
        gpf.Parameter with transform
    """
    return gpf.Parameter(data, name=data.name.split(":")[0], transform=transform,)


def extract_block_diag(array, block_size):
    """Extract block diagonals of square 2D Array.

    Args:
        array (np.ndarray): square 2D array
        block_size (int): block size

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
