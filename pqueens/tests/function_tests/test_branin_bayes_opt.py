from pqueens.main import main
import pytest
import pickle

def test_branin_bayes_opt(tmpdir):
    """ Test case for Bayesian optimization using a Gaussian process """
    arguments = ['--input=pqueens/tests/function_tests/input_files/branin_bayes_opt.json',
                 '--output='+str(tmpdir)]

    main(arguments)
    result_file = str(tmpdir)+'/'+'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)
        minimum = results["x"][0]
    # Assert that the found minimum is at -3.333 and 13.33
    assert minimum[0] == pytest.approx(-3.33333333)
    assert minimum[1] == pytest.approx(13.33333333)
