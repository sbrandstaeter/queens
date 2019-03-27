from pqueens.main import main
import pytest
import pickle

def test_branin_monte_carlo(tmpdir):
    """ Test case for monte carlo iterator """
    arguments = ['--input=pqueens/tests/function_tests/input_files/branin_monte_carlo.json',
                 '--output='+str(tmpdir)]

    main(arguments)
    result_file = str(tmpdir)+'/'+'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)
    assert results["mean"] == pytest.approx(55.81419875080866)
    assert results["var"] == pytest.approx(2754.1188056842070)
