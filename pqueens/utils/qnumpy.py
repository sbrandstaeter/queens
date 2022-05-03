"""Module for QArray data class."""

import numpy as np


class QArray:
    """Class for Data handling.

    This QArray class is a wrapper around numpy.

    Attributes:
        _arr (np.ndarray): Array containing data.
        _axes (dict): Dictionary containing labels for axes of _arr
    """

    def __init__(self, arr, axes):
        """Initialize the QArray object.

        Args:
            arr (np.ndarray): Array containing data.
            axes (dict): Dictionary containing labels for axes of arr
        """
        self._arr = arr
        self._axes = axes

    def to_numpy(self, order=None):
        """Convert a QArray to numpy array.

        Args:
            order (list, optional): List of keys to specify the order of numpy array axes

        Returns:
            np_arr (np.ndarray): Numpy array with specified or default ordering
        """
        if order is None:
            np_arr = self._arr
        else:
            order = list(order)
            if len(order) != len(self._axes):
                raise ValueError(
                    f'The required number of keys is {len(self._axes)}, but you only '
                    f'provided {len(order)} keys.'
                )
            destination = list(range(0, len(order)))
            source = [self._axes[ax_key]['index'] for ax_key in order]
            np_arr = np.moveaxis(self._arr, source, destination)
        return np_arr

    def return_axes(self):
        """Return axes dictionary.

        Returns:
            self._axes (dict): Dictionary containing labels for axes of self._arr
        """
        return self._axes

    def return_axes_keys(self):
        """Return axes keys.

        Returns:
            ax_keys (list): List containing keys of self._axes dictionary
        """
        ax_keys = self._axes.keys()
        return ax_keys

    def array_shape(self):
        """Return shape of array.

        Returns:
            shape (tuple): Returns the shape of the array
        """
        shape = self._arr.shape
        return shape

    def ax_indices_from_keys(self, keys):
        """Return the indices corresponding to specified keys.

        Args:
            keys (list): List of keys

        Returns:
            ax_keys (list): Array indices corresponding to specified keys
        """
        keys = list(keys)
        ax_keys = [self._axes[key]['index'] for key in keys]
        return ax_keys

    def ax_keys_from_indices(self, indices):
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
            for key in self._axes.keys()
            if self._axes[key]['index'] == index
        ]
        return ax_keys


def _create_unique_keys(keys):
    """Create unique keys.

    Rename duplicate keys by extending the name with _1, _2, ...

    Args:
        keys (list): List of keys

    Returns:
        unique_keys (list): List of keys with unique names
    """
    unique_keys = list(
        dict.fromkeys(
            [
                key if keys.count(key) == 1 else key + '_' + str(i + 1)
                for key in keys
                for i in range(keys.count(key))
            ]
        )
    )
    return unique_keys


def array(arr, ax_keys):
    """Create a QArray object.

    Args:
        arr (np.ndarray): Array containing data.
        ax_keys (list): List of strings containing labels for axes of arr

    Returns: q_arr (QArray): QArray object with the specified axes
    """
    shape = arr.shape
    axes = {}
    for i, ax_key in enumerate(ax_keys):
        axes[ax_key] = {}
        axes[ax_key]['index'] = i
        axes[ax_key]['len'] = shape[i]
    q_arr = QArray(arr, axes)
    return q_arr


def dot_product(arr1, arr2, axes):
    """Dot product / Tensor contraction.

    Args:
        arr1 (QArray): Array containing data
        arr2 (QArray): Array containing data
        axes (str, array_like): axis over which the inner product is evaluated

    Returns: q_arr (QArray): Dot product / Tensor contraction of the input arrays
    """
    np_arr1 = arr1.to_numpy()
    np_arr2 = arr2.to_numpy()
    product_indices_1 = arr1.ax_indices_from_keys(axes)
    product_indices_2 = arr2.ax_indices_from_keys(axes)
    all_indices_1 = list(range(0, np_arr1.ndim))
    all_indices_2 = list(range(0, np_arr2.ndim))
    remaining_indices_1 = [i for i in all_indices_1 if i not in product_indices_1]
    remaining_indices_2 = [i for i in all_indices_2 if i not in product_indices_2]
    remaining_keys_1 = arr1.ax_keys_from_indices(remaining_indices_1)
    remaining_keys_2 = arr2.ax_keys_from_indices(remaining_indices_2)
    res_arr = np.tensordot(np_arr1, np_arr2, (product_indices_1, product_indices_2))
    res_axes = _create_unique_keys(remaining_keys_1 + remaining_keys_2)
    q_arr = array(res_arr, res_axes)
    return q_arr
