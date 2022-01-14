import os

import pytest

from pqueens.main import main


@pytest.mark.integration_tests
def test_rosenbrock_lsq_opt_error(inputdir, tmpdir):
    """Test case for optimization iterator with least squares."""
    arguments = [
        '--input=' + os.path.join(inputdir, 'rosenbrock_lsq_opt_error.json'),
        '--output=' + str(tmpdir),
    ]

    with pytest.raises(ValueError):
        main(arguments)
