import os
import pickle

import numpy as np
import pytest

from pqueens.main import main


@pytest.mark.integration_tests
def test_branin_data_iterator(inputdir, tmpdir, mocker):
    """Test case for data iterator."""
    arguments = [
        '--input=' + os.path.join(inputdir, 'branin_data_iterator.json'),
        '--output=' + str(tmpdir),
    ]

    output = {}
    output['mean'] = np.array(
        [
            [1.7868040337],
            [-13.8624183835],
            [6.3423271929],
            [6.1674472752],
            [5.3528917433],
            [-0.7472766806],
            [5.0007066283],
            [6.4763926539],
            [-6.4173504897],
            [3.1739282221],
        ]
    )

    samples = np.array(
        [
            [1.7868040337],
            [-13.8624183835],
            [6.3423271929],
            [6.1674472752],
            [5.3528917433],
            [-0.7472766806],
            [5.0007066283],
            [6.4763926539],
            [-6.4173504897],
            [3.1739282221],
        ]
    )

    mocker.patch(
        'pqueens.iterators.data_iterator.DataIterator.read_pickle_file',
        return_value=[samples, output],
    )

    main(arguments)
    result_file = str(tmpdir) + '/' + 'xxx.pickle'
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)
    assert results["mean"] == pytest.approx(1.3273452195599997)
    assert results["var"] == pytest.approx(44.82468751096612)
