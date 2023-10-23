"""Module supplies function for encoding NumPy array into JSON."""

from json import JSONEncoder

import numpy as np


class NumpyArrayEncoder(JSONEncoder):
    """Numpy array encoder.

    Based on the JSONEncoder object.
    """

    def default(self, o):
        """Encode np.array.

        Either to int, float or list.

        Args:
            o (obj): Object to encode.

        Returns:
            TODO_doc
        """
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)
