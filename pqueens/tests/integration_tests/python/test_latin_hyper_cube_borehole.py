"""TODO_doc."""

import pickle

import pytest

from pqueens import run


def test_latin_hyper_cube_borehole(inputdir, tmp_path):
    """Test case for latin hyper cube iterator."""
    run(inputdir / 'latin_hyper_cube_borehole.yml', tmp_path)

    result_file = tmp_path / 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)
    assert results["mean"] == pytest.approx(62.05240444441511)
    assert results["var"] == pytest.approx(1371.7554224384000)
