import os

import pytest

from pqueens.main import main


@pytest.mark.integration_tests
def test_optimization_lsq_rosenbrock_error(inputdir, tmpdir):
    """Test case for optimization iterator with least squares."""
    arguments = [
        '--input=' + os.path.join(inputdir, 'optimization_lsq_rosenbrock_error.json'),
        '--output=' + str(tmpdir),
    ]

    with pytest.raises(TypeError, match="got an unexpected keyword argument 'x3'"):
        main(arguments)
