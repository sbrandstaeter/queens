import os
import pickle

import pytest

from pqueens.main import main


@pytest.mark.integration_tests
def test_sobol_indices_ishigami(inputdir, tmpdir):
    """Test case for salib based saltelli iterator."""
    arguments = [
        '--input=' + os.path.join(inputdir, 'sobol_indices_ishigami.json'),
        '--output=' + str(tmpdir),
    ]

    main(arguments)
    result_file = str(tmpdir) + '/' + 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    assert results["sensitivity_indices"]['S1'][0] == pytest.approx(-0.20343406)
    assert results["sensitivity_indices"]['S1'][1] == pytest.approx(0.83832789)
    assert results["sensitivity_indices"]['S1'][2] == pytest.approx(-0.59208696)

    assert results["sensitivity_indices"]['ST'][0] == pytest.approx(1.34583705)
    assert results["sensitivity_indices"]['ST'][1] == pytest.approx(0.71692249)
    assert results["sensitivity_indices"]['ST'][2] == pytest.approx(0.45987242)

    assert results["sensitivity_indices"]['S2'][0, 1] == pytest.approx(0.20866473)
    assert results["sensitivity_indices"]['S2'][0, 2] == pytest.approx(1.16521165)
    assert results["sensitivity_indices"]['S2'][1, 2] == pytest.approx(0.2685581)

    assert results["sensitivity_indices"]['S1_conf'][0] == pytest.approx(0.67773088)
    assert results["sensitivity_indices"]['S1_conf'][1] == pytest.approx(0.60778569)
    assert results["sensitivity_indices"]['S1_conf'][2] == pytest.approx(0.37909073)

    assert results["sensitivity_indices"]['ST_conf'][0] == pytest.approx(2.21068444)
    assert results["sensitivity_indices"]['ST_conf'][1] == pytest.approx(0.55758642)
    assert results["sensitivity_indices"]['ST_conf'][2] == pytest.approx(0.41049134)

    assert results["sensitivity_indices"]['S2_conf'][0, 1] == pytest.approx(1.19398233)
    assert results["sensitivity_indices"]['S2_conf'][0, 2] == pytest.approx(0.87755398)
    assert results["sensitivity_indices"]['S2_conf'][1, 2] == pytest.approx(0.76481440)
