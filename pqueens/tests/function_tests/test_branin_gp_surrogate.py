from pqueens.main import main
import pytest
import pickle

def test_branin_gp_surrogate(tmpdir):
    """ Test case for GP based surrogate model """
    arguments = ['--input=pqueens/tests/function_tests/input_files/branin_gp_surrogate.json',
                 '--output='+str(tmpdir)]

    main(arguments)
    result_file = str(tmpdir)+'/'+'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)
    assert results["pdf_estimate"]["mean"][1] == pytest.approx(8.860833411995653e-05, rel=1e-3)
