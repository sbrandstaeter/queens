"""Module supplies function for encoding NumPy array into JSON."""

import json
from json import JSONEncoder

import numpy as np


class NumpyArrayEncoder(JSONEncoder):
    """Numpy array encoder.

    Based on the JSONEncoder object.
    """

    def default(self, obj):
        """Encode np.array.

        Either to int, float or list.

        Args:
            obj (obj): Object to encode.
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super().default(obj)
