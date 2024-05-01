"""TODO_doc."""

import pytest

from queens.main import run
from queens.utils.io_utils import load_result


def test_branin_monte_carlo(inputdir, tmp_path):
    """Test case for Monte Carlo iterator."""
    run(inputdir / 'monte_carlo_branin.yml', tmp_path)

    results = load_result(tmp_path / 'xxx.pickle')
    assert results["mean"] == pytest.approx(55.81419875080866)
    assert results["var"] == pytest.approx(2754.1188056842070)
