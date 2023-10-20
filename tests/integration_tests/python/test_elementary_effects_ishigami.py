"""TODO_doc."""

import logging
import pickle

import pytest

from queens.main import run

_logger = logging.getLogger(__name__)


def test_elementary_effects_ishigami(inputdir, tmp_path):
    """Test case for elementary effects iterator."""
    run(inputdir / 'elementary_effects_ishigami.yml', tmp_path)

    result_file = tmp_path / 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    _logger.info(results)

    assert results["sensitivity_indices"]['mu'][0] == pytest.approx(15.46038594, abs=1e-7)
    assert results["sensitivity_indices"]['mu'][1] == pytest.approx(0.0, abs=1e-7)
    assert results["sensitivity_indices"]['mu'][2] == pytest.approx(0.0, abs=1e-7)

    assert results["sensitivity_indices"]['mu_star'][0] == pytest.approx(15.460385940, abs=1e-7)
    assert results["sensitivity_indices"]['mu_star'][1] == pytest.approx(1.47392000, abs=1e-7)
    assert results["sensitivity_indices"]['mu_star'][2] == pytest.approx(5.63434321, abs=1e-7)

    assert results["sensitivity_indices"]['sigma'][0] == pytest.approx(15.85512257, abs=1e-7)
    assert results["sensitivity_indices"]['sigma'][1] == pytest.approx(1.70193622, abs=1e-7)
    assert results["sensitivity_indices"]['sigma'][2] == pytest.approx(9.20084394, abs=1e-7)

    assert results["sensitivity_indices"]['mu_star_conf'][0] == pytest.approx(13.53414548, abs=1e-7)
    assert results["sensitivity_indices"]['mu_star_conf'][1] == pytest.approx(0.0, abs=1e-7)
    assert results["sensitivity_indices"]['mu_star_conf'][2] == pytest.approx(5.51108773, abs=1e-7)
