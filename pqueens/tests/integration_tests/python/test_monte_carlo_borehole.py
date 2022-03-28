import os
import pickle

import pytest

from pqueens.main import main


@pytest.mark.integration_tests
def test_monte_carlo_borehole(inputdir, tmpdir):
    """Test case for monte carlo iterator."""
    arguments = [
        '--input=' + os.path.join(inputdir, 'monte_carlo_borehole.json'),
        '--output=' + str(tmpdir),
    ]

    main(arguments)
    result_file = str(tmpdir) + '/' + 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)
    assert results["mean"] == pytest.approx(60.4546131041304)
    assert results["var"] == pytest.approx(1268.1681250046817)
