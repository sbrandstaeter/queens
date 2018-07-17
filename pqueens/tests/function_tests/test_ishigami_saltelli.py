from pqueens.main import main
import pytest
import pickle

def test_ishigami_saltelli_salib(tmpdir):
    """ Test case for salib based saltelli iterator """
    arguments = ['--input=pqueens/tests/function_tests/input_files/ishigami_saltelli.json',
                 '--output='+str(tmpdir)]

    main(arguments)
    result_file = str(tmpdir)+'/'+'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    #rint(results)

    assert results["sensitivity_incides"]['S1'][0] == pytest.approx(-0.20343406)
    assert results["sensitivity_incides"]['S1'][1] == pytest.approx(0.83832789)
    assert results["sensitivity_incides"]['S1'][2] == pytest.approx(-0.59208696)

    assert results["sensitivity_incides"]['S1_conf'][0] == pytest.approx(0.69122846)
    assert results["sensitivity_incides"]['S1_conf'][1] == pytest.approx(0.62588678)
    assert results["sensitivity_incides"]['S1_conf'][2] == pytest.approx(0.37978581)

    assert results["sensitivity_incides"]['ST'][0] == pytest.approx(1.34583705)
    assert results["sensitivity_incides"]['ST'][1] == pytest.approx(0.71692249)
    assert results["sensitivity_incides"]['ST'][2] == pytest.approx(0.45987242)

    assert results["sensitivity_incides"]['ST_conf'][0] == pytest.approx(2.2341512)
    assert results["sensitivity_incides"]['ST_conf'][1] == pytest.approx(0.64054538)
    assert results["sensitivity_incides"]['ST_conf'][2] == pytest.approx(0.42364799)

    assert results["sensitivity_incides"]['S2'][0, 1] == pytest.approx(0.20866473)
    assert results["sensitivity_incides"]['S2'][0, 2] == pytest.approx(1.16521165)
    assert results["sensitivity_incides"]['S2'][1, 2] == pytest.approx(0.2685581)

    assert results["sensitivity_incides"]['S2_conf'][0, 1] == pytest.approx(1.22303931)
    assert results["sensitivity_incides"]['S2_conf'][0, 2] == pytest.approx(0.92533686)
    assert results["sensitivity_incides"]['S2_conf'][1, 2] == pytest.approx(0.7434732)
