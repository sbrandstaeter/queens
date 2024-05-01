"""TODO_doc."""

import pytest

from queens.main import run
from queens.utils.io_utils import load_result


def test_latin_hyper_cube_borehole(inputdir, tmp_path):
    """Test case for latin hyper cube iterator."""
    run(inputdir / 'latin_hyper_cube_borehole.yml', tmp_path)

    results = load_result(tmp_path / 'xxx.pickle')
    assert results["mean"] == pytest.approx(62.05240444441511)
    assert results["var"] == pytest.approx(1371.7554224384000)
