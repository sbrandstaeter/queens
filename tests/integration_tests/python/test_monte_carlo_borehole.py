"""TODO_doc."""

import pytest

from queens.main import run
from queens.utils.io_utils import load_result


def test_monte_carlo_borehole(inputdir, tmp_path):
    """Test case for Monte Carlo iterator."""
    run(inputdir / 'monte_carlo_borehole.yml', tmp_path)

    results = load_result(tmp_path / 'xxx.pickle')
    assert results["mean"] == pytest.approx(60.4546131041304)
    assert results["var"] == pytest.approx(1268.1681250046817)
