"""Test chaospy wrapper."""
import os
import pickle

import pytest

from pqueens.main import main


@pytest.mark.integration_tests
def test_polynomial_chaos_pseudo_spectral_borehole(inputdir, tmpdir):
    """Test case for the pc iterator using a pseudo spectral approach."""
    arguments = [
        '--input=' + os.path.join(inputdir, 'polynomial_chaos_pseudo_spectral_borehole.json'),
        '--output=' + str(tmpdir),
    ]

    main(arguments)
    result_file = str(tmpdir) + '/' + 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)
    assert results["mean"] == pytest.approx(61.78966587)
    assert results["covariance"] == pytest.approx([1312.23414971])


@pytest.mark.integration_tests
def test_polynomial_chaos_collocation_borehole(inputdir, tmpdir):
    """Test for the pc iterator using a collocation approach."""
    arguments = [
        '--input=' + os.path.join(inputdir, 'polynomial_chaos_collocation_borehole.json'),
        '--output=' + str(tmpdir),
    ]

    main(arguments)
    result_file = str(tmpdir) + '/' + 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)
    assert results["mean"] == pytest.approx(62.05018243)
    assert results["covariance"] == pytest.approx([1273.81372103])
