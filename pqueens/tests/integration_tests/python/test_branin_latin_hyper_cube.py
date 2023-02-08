"""TODO_doc."""

import pickle
from pathlib import Path

import pytest

from pqueens import run


def test_branin_latin_hyper_cube(inputdir, tmp_path):
    """Test case for latin hyper cube iterator."""
    run(inputdir / 'latin_hyper_cube_branin.yml', tmp_path)

    result_file = tmp_path / 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)
    assert results["mean"] == pytest.approx(53.17279969296224)
    assert results["var"] == pytest.approx(2581.6502630157715)
