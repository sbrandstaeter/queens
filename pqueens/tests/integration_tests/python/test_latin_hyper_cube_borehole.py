import os
import pickle
from pathlib import Path

import pytest

from pqueens import run


@pytest.mark.integration_tests
def test_latin_hyper_cube_borehole(inputdir, tmpdir):
    """Test case for latin hyper cube iterator."""
    run(Path(os.path.join(inputdir, 'latin_hyper_cube_borehole.json')), Path(tmpdir))

    result_file = str(tmpdir) + '/' + 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)
    assert results["mean"] == pytest.approx(62.05240444441511)
    assert results["var"] == pytest.approx(1371.7554224384000)
