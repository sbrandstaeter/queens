"""TODO_doc."""

import pickle

import pytest

from queens.main import run


def test_monte_carlo_borehole(inputdir, tmp_path):
    """Test case for Monte Carlo iterator."""
    run(inputdir / 'monte_carlo_borehole.yml', tmp_path)

    result_file = tmp_path / 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)
    assert results["mean"] == pytest.approx(60.4546131041304)
    assert results["var"] == pytest.approx(1268.1681250046817)
