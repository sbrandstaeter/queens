import os
import pickle

import pytest

from pqueens.main import main


def test_borehole_monte_carlo(inputdir, tmpdir):
    """ Test case for monte carlo iterator """
    arguments = [
        '--input=' + os.path.join(inputdir, 'borehole_monte_carlo.json'),
        '--output=' + str(tmpdir),
    ]

    main(arguments)
    result_file = str(tmpdir) + '/' + 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)
    assert results["mean"] == pytest.approx(60.4546131041304)
    assert results["var"] == pytest.approx(1268.1681250046817)
