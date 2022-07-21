import os
import pickle
from pathlib import Path

import pytest

from pqueens import run


@pytest.mark.integration_tests
def test_monte_carlo_borehole(inputdir, tmpdir):
    """Test case for monte carlo iterator."""
    run(Path(os.path.join(inputdir, 'monte_carlo_borehole.json')), Path(tmpdir))

    result_file = str(tmpdir) + '/' + 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)
    assert results["mean"] == pytest.approx(60.4546131041304)
    assert results["var"] == pytest.approx(1268.1681250046817)
