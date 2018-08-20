from pqueens.main import main
import pytest
import pickle

def test_borehole_saltelli(tmpdir):
    """ Test case for SA_lib saltelli iterator """
    arguments = ['--input=pqueens/tests/function_tests/input_files/borehole_salib_saltelli.json',
                 '--output='+str(tmpdir)]

    main(arguments)
    result_file = str(tmpdir)+'/'+'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)
    #print("results {}".format(results))
    #print("Parameter Names {}".format(results["parameter_names"]))
    #print("sensitivity_incides][S1] {}".format(results["sensitivity_incides"]["S1"]))

    assert results["sensitivity_incides"]["S1"][0] == pytest.approx(8.26790823e-01)
    assert results["sensitivity_incides"]["S1"][1] == pytest.approx(7.27449652e-06)
    assert results["sensitivity_incides"]["S1"][2] == pytest.approx(-4.92805789e-09)
    assert results["sensitivity_incides"]["S1"][3] == pytest.approx(4.14903497e-02)
    assert results["sensitivity_incides"]["S1"][4] == pytest.approx(1.60534877e-05)
    assert results["sensitivity_incides"]["S1"][5] == pytest.approx(4.17984899e-02)
    assert results["sensitivity_incides"]["S1"][6] == pytest.approx(3.94960542e-02)
    assert results["sensitivity_incides"]["S1"][7] == pytest.approx(9.63445696e-03)
