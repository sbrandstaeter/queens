"""Module for QArray data class."""

import numpy as np


class QArray:
    """Class for Data handling.

    This QArray class is a wrapper around numpy.

    Attributes:
        _array (np.ndarray): Array containing data.
        _array_struct (dict): Dictionary containing labels for axes of _array
    """

    def __init__(self, np_arr, array_struct):
        """Initialize the QArray object.

        Args:
            np_arr (np.ndarray): Array containing data.
            array_struct (dict): Dictionary containing labels for axes of numpy array
        """
        self._array = np_arr
        self._array_struct = array_struct

    def to_numpy(self, order=None):
        """Convert a QArray to numpy array.

        Args:
            order (list, optional): List of keys to specify the order of numpy array axes

        Returns:
            np_arr (np.ndarray): Numpy array with specified or default ordering
        """
        if order is None:
            np_arr = self._array
        else:
            order = list(order)
            if len(order) != len(self._array_struct):
                raise ValueError(
                    f'The required number of keys is {len(self._array_struct)}, but '
                    f'{len(order)} keys were provided.'
                )
            destination = list(range(0, len(order)))
            source = [self._array_struct[ax_key]['index'] for ax_key in order]
            np_arr = np.moveaxis(self._array, source, destination)
        return np_arr

    def return_array_struct(self):
        """Return array structure.

        Returns:
            self._array_struct (dict): Dictionary containing labels for axes of self._array
        """
        return self._array_struct

    def return_axes_keys(self):
        """Return axes keys.

        Returns:
            ax_keys (list): List containing keys of self._array_struct dictionary
        """
        ax_keys = self._array_struct.keys()
        return ax_keys

    def array_shape(self):
        """Return shape of array.

        Returns:
            shape (tuple): Returns the shape of the array
        """
        shape = self._array.shape
        return shape

    def axes_indices_from_keys(self, keys):
        """Return the indices corresponding to specified keys.

        Args:
            keys (list): List of keys

        Returns:
            ax_keys (list): Array indices corresponding to specified keys
        """
        keys = list(keys)
        ax_keys = [self._array_struct[key]['index'] for key in keys]
        return ax_keys

    def axes_keys_from_indices(self, indices):
        """Return the keys corresponding to specified indices.

        Args:
            indices (list): List of indices

        Returns:
            ax_keys (list): Keys corresponding to specified array indices
        """
        indices = list(indices)
        ax_keys = [
            key
            for index in indices
            for key in self._array_struct.keys()
            if self._array_struct[key]['index'] == index
        ]
        return ax_keys


def array(arr, ax_keys):
    """Create a QArray object.

    Args:
        arr (np.ndarray): Array containing data.
        ax_keys (list): List of strings containing labels for axes of arr

    Returns: q_arr (QArray): QArray object with the specified axes
    """
    ax_keys = list(ax_keys)
    if len(ax_keys) > len(set(ax_keys)):
        raise ValueError('Provided list of axes keys must be unique!')

    shape = arr.shape
    axes = {}
    for i, ax_key in enumerate(ax_keys):
        axes[ax_key] = {}
        axes[ax_key]['index'] = i
        axes[ax_key]['len'] = shape[i]
    q_arr = QArray(arr, axes)
    return q_arr


def tensordot(arr1, arr2, product_keys, result_keys):
    """Compute tensor dot product along specified axes.

    Args:
        arr1 (QArray): Array containing data
        arr2 (QArray): Array containing data
        product_keys (str, array_like): axes over which the inner product is evaluated
        result_keys (str, array_like): Axes keys of the resulting array

    Returns: q_arr (QArray): Computed tensor dot product along specified axes of the input arrays
    """
    np_arr1 = arr1.to_numpy()
    np_arr2 = arr2.to_numpy()
    product_indices_1 = arr1.axes_indices_from_keys(product_keys)
    product_indices_2 = arr2.axes_indices_from_keys(product_keys)
    result_array = np.tensordot(np_arr1, np_arr2, (product_indices_1, product_indices_2))
    q_arr = array(result_array, result_keys)
    return q_arr
