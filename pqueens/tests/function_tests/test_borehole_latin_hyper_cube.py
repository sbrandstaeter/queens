import os
import pickle

import pytest

from pqueens.main import main


def test_borehole_latin_hyper_cube(inputdir, tmpdir):
    """ Test case for latin hyper cube iterator """
    arguments = [
        '--input=' + os.path.join(inputdir, 'borehole_latin_hyper_cube.json'),
        '--output=' + str(tmpdir),
    ]

    main(arguments)
    result_file = str(tmpdir) + '/' + 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)
    assert results["mean"] == pytest.approx(62.05240444441511)
    assert results["var"] == pytest.approx(1371.7554224384000)
