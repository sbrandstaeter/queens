import pickle

import numpy as np
import pytest

from pqueens.main import main


def test_rosenbrock_lsq_opt_error(tmpdir):
    """ Test case for optimization iterator with least squares. """
    arguments = ['--input=pqueens/tests/function_tests/input_files/rosenbrock_lsq_opt_error.json',
                 '--output='+str(tmpdir)]

    with pytest.raises(ValueError):
        main(arguments)
