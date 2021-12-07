import os
import pickle

import pytest

from pqueens.main import main


@pytest.mark.integration_tests
def test_branin_monte_carlo(inputdir, tmpdir):
    """Test case for monte carlo iterator."""
    arguments = [
        '--input=' + os.path.join(inputdir, 'branin_monte_carlo.json'),
        '--output=' + str(tmpdir),
    ]

    main(arguments)
    result_file = str(tmpdir) + '/' + 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)
    assert results["mean"] == pytest.approx(55.81419875080866)
    assert results["var"] == pytest.approx(2754.1188056842070)
