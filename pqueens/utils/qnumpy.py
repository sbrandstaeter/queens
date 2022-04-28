"""Module for QArray data class."""

import numpy as np


class QArray:
    """Class for Data handling.

    This QArray class is a wrapper around numpy.

    Attributes:
        _arr (np.ndarray): Array containing data.
        _axes (dict): Dictionary containing labels for axes of _arr
    """

    def __init__(self, arr, ax_keys):
        """Initialize iterator object.

        Args:
            arr (np.ndarray): Array containing data.
            ax_keys (list): List of strings containing labels for axes of arr
        """
        self._arr = arr
        self._axes = {}
        shape = arr.shape
        ax_keys = self._create_unique_keys(ax_keys)
        for i, ax_key in enumerate(ax_keys):
            self._axes[ax_key] = {}
            self._axes[ax_key]['index'] = i
            self._axes[ax_key]['len'] = shape[i]

    def to_numpy(self, order=None):
        """Return np.ndarray.

        Args:
            order (list): List of keys to specify the order of numpy array axes (optional)

        Returns:
            np.ndarray: Numpy array with specified or default ordering
        """
        if order is None:
            return self._arr
        else:
            if len(order) != len(self._axes):
                raise ValueError('Specified order does not match the number of dimensions!')
            destination = list(range(0, len(order)))
            source = [self._axes[ax_key]['index'] for ax_key in order]
            return np.moveaxis(self._arr, source, destination)

    def axes(self):
        """Return axis dictionary.

        Returns:
            self._axes (dict): Dictionary containing labels for axes of self._arr
        """
        return self._axes

    def axes_keys(self):
        """Return axes keys.

        Returns:
            list: List containing keys of self._axes dictionary
        """
        return self._axes.keys()

    def array_shape(self):
        """Return shape of array.

        Returns:
            tuple: Returns the shape of the array
        """
        return self._arr.shape

    def ax_indices_from_keys(self, keys):
        """Return the indices corresponding to specified keys.

        Args:
            keys (list): List of keys

        Returns:
            list: Array indices corresponding to specified keys
        """
        keys = list(keys)
        return [self._axes[key]['index'] for key in keys]

    def ax_keys_from_indices(self, indices):
        """Return the keys corresponding to specified indices.

        Args:
            indices (list): List of indices

        Returns:
            list: Keys corresponding to specified array indices
        """
        indices = list(indices)
        return [
            key
            for index in indices
            for key in self._axes.keys()
            if self._axes[key]['index'] == index
        ]

    @staticmethod
    def _create_unique_keys(keys):
        """Create unique keys.

        Rename duplicate keys by extending the name with _1, _2, ...

        Args:
            keys (list): List of keys

        Returns:
            list: List of keys with unique names
        """
        return list(
            dict.fromkeys(
                [
                    key if keys.count(key) == 1 else key + '_' + str(i + 1)
                    for key in keys
                    for i in range(keys.count(key))
                ]
            )
        )


def dot_product(arr1, arr2, axes):
    """Dot product / Tensor contraction.

    Args:
        arr1 (QArray): Array containing data
        arr2 (QArray): Array containing data
        axes (str, array_like): axis over which the inner product is evaluated

    Returns: QArray: Dot product / Tensor contraction of the input arrays
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
    res_axes = remaining_keys_1 + remaining_keys_2

    return QArray(res_arr, res_axes)
