from pqueens.main import main
import pytest
import pickle

def test_borehole_monte_carlo(tmpdir):
    """ Test case for monte carlo iterator """
    arguments = ['--input=pqueens/tests/function_tests/input_files/borehole_monte_carlo.json',
                 '--output='+str(tmpdir)]

    main(arguments)
    result_file = str(tmpdir)+'/'+'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)
    assert results["mean"] == pytest.approx(60.4546131041304)
    assert results["var"] == pytest.approx(1266.8999568796771)
