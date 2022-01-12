"""Test Sobol indices estimation for borehole function."""
import os
import pickle

import pytest

from pqueens.main import main


@pytest.mark.integration_tests
def test_sobol_indices_borehole(inputdir, tmpdir):
    """Test case for Sobol Index iterator."""
    arguments = [
        '--input=' + os.path.join(inputdir, 'sobol_indices_borehole.json'),
        '--output=' + str(tmpdir),
    ]

    main(arguments)
    result_file = str(tmpdir) + '/' + 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    assert results["sensitivity_indices"]["S1"][0] == pytest.approx(0.8275788005095177)
    assert results["sensitivity_indices"]["S1"][1] == pytest.approx(3.626326582692376e-05)
    assert results["sensitivity_indices"]["S1"][2] == pytest.approx(1.7993448562887368e-09)
    assert results["sensitivity_indices"]["S1"][3] == pytest.approx(0.04082350205109995)
    assert results["sensitivity_indices"]["S1"][4] == pytest.approx(-1.0853339811788176e-05)
    assert results["sensitivity_indices"]["S1"][5] == pytest.approx(0.0427473897346278)
    assert results["sensitivity_indices"]["S1"][6] == pytest.approx(0.038941629762778956)
    assert results["sensitivity_indices"]["S1"][7] == pytest.approx(0.009001905983634081)
