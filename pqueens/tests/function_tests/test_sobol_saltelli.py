import os
import pickle

import pytest

from pqueens.main import main


def test_sobol_saltelli(inputdir, tmpdir):
    """ Test case for saltelli iterator """
    arguments = [
        '--input=' + os.path.join(inputdir, 'sobol_saltelli.json'),
        '--output=' + str(tmpdir),
    ]

    main(arguments)
    result_file = str(tmpdir) + '/' + 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    assert results["sensitivity_incides"]["S1"][0] == pytest.approx(7.33626148e-01)
    assert results["sensitivity_incides"]["S1"][1] == pytest.approx(1.75758359e-01)
    assert results["sensitivity_incides"]["S1"][2] == pytest.approx(2.24732283e-02)
    assert results["sensitivity_incides"]["S1"][3] == pytest.approx(7.58282827e-03)
    assert results["sensitivity_incides"]["S1"][4] == pytest.approx(-5.69836812e-05)
    assert results["sensitivity_incides"]["S1"][5] == pytest.approx(2.42598472e-04)
    assert results["sensitivity_incides"]["S1"][6] == pytest.approx(2.92864865e-04)
    assert results["sensitivity_incides"]["S1"][7] == pytest.approx(3.68325749e-05)
