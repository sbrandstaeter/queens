import os
import pickle
from pathlib import Path

import pytest

from pqueens import run


@pytest.mark.integration_tests
def test_elementary_effects_ishigami(inputdir, tmpdir):
    """Test case for elementary effects iterator."""
    run(Path(os.path.join(inputdir, 'elementary_effects_ishigami.json')), Path(tmpdir))

    result_file = str(tmpdir) + '/' + 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    print(results)

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
