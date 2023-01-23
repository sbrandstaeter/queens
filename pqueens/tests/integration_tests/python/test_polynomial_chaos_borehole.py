"""Test chaospy wrapper."""
import os
import pickle
from pathlib import Path

import pytest

from pqueens import run


def test_polynomial_chaos_pseudo_spectral_borehole(inputdir, tmpdir):
    """Test case for the PC iterator using a pseudo spectral approach."""
    run(Path(Path(inputdir, 'polynomial_chaos_pseudo_spectral_borehole.yml')), Path(tmpdir))

    result_file = str(tmpdir) + '/' + 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)
    assert results["mean"] == pytest.approx(61.78966587)
    assert results["covariance"] == pytest.approx([1312.23414971])


def test_polynomial_chaos_collocation_borehole(inputdir, tmpdir):
    """Test for the PC iterator using a collocation approach."""
    run(Path(Path(inputdir, 'polynomial_chaos_collocation_borehole.yml')), Path(tmpdir))

    result_file = str(tmpdir) + '/' + 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)
    assert results["mean"] == pytest.approx(62.05018243)
    assert results["covariance"] == pytest.approx([1273.81372103])
