from pqueens.main import main
import pytest
import pickle

def test_gaussian_metropolis_hastings(tmpdir):
    """ Test case for metropolis hastings iterator """
    arguments = ['--input=pqueens/tests/function_tests/input_files/gaussian_smc.json',
                 '--output='+str(tmpdir)]

    main(arguments)
    result_file = str(tmpdir)+'/'+'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    # note that the analytical solution would be:
    # posterior mean: [1.]
    # posterior var: [0.5]
    # posterior std: [0.70710678]
    # however, we only have a very inaccurate approximation here:
    assert results['mean'] == pytest.approx(1.7252416454013368)
    assert results['var'] == pytest.approx(1.1142160157656096)

