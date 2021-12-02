import os
import pickle

import pytest

from pqueens.main import main


@pytest.mark.integration_tests
def test_branin_latin_hyper_cube(inputdir, tmpdir):
    """ Test case for latin hyper cube iterator """
    arguments = [
        '--input=' + os.path.join(inputdir, 'branin_latin_hyper_cube.json'),
        '--output=' + str(tmpdir),
    ]

    main(arguments)
    result_file = str(tmpdir) + '/' + 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)
    assert results["mean"] == pytest.approx(53.17279969296224)
    assert results["var"] == pytest.approx(2581.6502630157715)
