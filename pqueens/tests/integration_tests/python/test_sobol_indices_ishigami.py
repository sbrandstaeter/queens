"""Test Sobol indices estimation for Ishigami function."""
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

    assert results["sensitivity_indices"]['S1'][0] == pytest.approx(0.12572757495660558)
    assert results["sensitivity_indices"]['S1'][1] == pytest.approx(0.3888444532476749)
    assert results["sensitivity_indices"]['S1'][2] == pytest.approx(-0.1701023677236496)

    assert results["sensitivity_indices"]['S1_conf'][0] == pytest.approx(0.3935803586836114)
    assert results["sensitivity_indices"]['S1_conf'][1] == pytest.approx(0.6623091120357786)
    assert results["sensitivity_indices"]['S1_conf'][2] == pytest.approx(0.2372589075839736)

    assert results["sensitivity_indices"]['ST'][0] == pytest.approx(0.32520201992825987)
    assert results["sensitivity_indices"]['ST'][1] == pytest.approx(0.5263552164769918)
    assert results["sensitivity_indices"]['ST'][2] == pytest.approx(0.1289289258091274)

    assert results["sensitivity_indices"]['ST_conf'][0] == pytest.approx(0.24575185898081872)
    assert results["sensitivity_indices"]['ST_conf'][1] == pytest.approx(0.5535870474744364)
    assert results["sensitivity_indices"]['ST_conf'][2] == pytest.approx(0.15792828597131078)

    assert results["sensitivity_indices"]['S2'][0, 1] == pytest.approx(0.6350854922111611)
    assert results["sensitivity_indices"]['S2'][0, 2] == pytest.approx(1.0749774123116016)
    assert results["sensitivity_indices"]['S2'][1, 2] == pytest.approx(0.32907368546743065)

    assert results["sensitivity_indices"]['S2_conf'][0, 1] == pytest.approx(0.840605849268133)
    assert results["sensitivity_indices"]['S2_conf'][0, 2] == pytest.approx(1.2064077218919202)
    assert results["sensitivity_indices"]['S2_conf'][1, 2] == pytest.approx(0.5803799668636836)
