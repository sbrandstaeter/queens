"""Test chaospy wrapper."""
import pickle

import pytest

from queens.main import run


def test_polynomial_chaos_pseudo_spectral_borehole(inputdir, tmp_path):
    """Test case for the PC iterator using a pseudo spectral approach."""
    run(inputdir / 'polynomial_chaos_pseudo_spectral_borehole.yml', tmp_path)

    result_file = tmp_path / "xxx.pickle"
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)
    assert results["mean"] == pytest.approx(61.78966587)
    assert results["covariance"] == pytest.approx([1312.23414971])


def test_polynomial_chaos_collocation_borehole(inputdir, tmp_path):
    """Test for the PC iterator using a collocation approach."""
    run(inputdir / 'polynomial_chaos_collocation_borehole.yml', tmp_path)

    result_file = tmp_path / 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)
    assert results["mean"] == pytest.approx(62.05018243)
    assert results["covariance"] == pytest.approx([1273.81372103])
