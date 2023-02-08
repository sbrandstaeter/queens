"""TODO_doc."""


import pickle
from pathlib import Path

import pytest

from pqueens import run


def test_branin_monte_carlo(inputdir, tmp_path):
    """Test case for Monte Carlo iterator."""
    run(inputdir / 'monte_carlo_branin.yml', tmp_path)

    result_file = tmp_path / 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)
    assert results["mean"] == pytest.approx(55.81419875080866)
    assert results["var"] == pytest.approx(2754.1188056842070)
