"""TODO_doc."""

import numpy as np
import pytest

from queens.main import run
from queens.utils.io_utils import load_result


def test_branin_data_iterator(inputdir, tmp_path, mocker):
    """Test case for data iterator."""
    output = {}
    output['result'] = np.array(
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
        'queens.iterators.data_iterator.DataIterator.read_pickle_file',
        return_value=[samples, output],
    )

    run(inputdir / 'data_iterator_branin.yml', tmp_path)
    results = load_result(tmp_path / 'xxx.pickle')
    assert results["mean"] == pytest.approx(1.3273452195599997)
    assert results["var"] == pytest.approx(44.82468751096612)
