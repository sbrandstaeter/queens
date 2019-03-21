from pqueens.main import main
import pytest
import pickle

def test_borehole_latin_hyper_cube(tmpdir):
    """ Test case for latin hyper cube iterator """
    arguments = ['--input=pqueens/tests/function_tests/input_files/borehole_latin_hyper_cube.json',
                 '--output='+str(tmpdir)]

    main(arguments)
    result_file = str(tmpdir)+'/'+'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)
    assert results["mean"] == pytest.approx(62.05240444441511)
    assert results["var"] == pytest.approx(1371.7554224384000)
