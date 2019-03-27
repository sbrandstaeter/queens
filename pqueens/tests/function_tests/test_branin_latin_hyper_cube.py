from pqueens.main import main
import pytest
import pickle

def test_branin_latin_hyper_cube(tmpdir):
    """ Test case for latin hyper cube iterator """
    arguments = ['--input=pqueens/tests/function_tests/input_files/branin_latin_hyper_cube.json',
                 '--output='+str(tmpdir)]

    main(arguments)
    result_file = str(tmpdir)+'/'+'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)
    assert results["mean"] == pytest.approx(53.17279969296224)
    assert results["var"] == pytest.approx(2581.6502630157715)
