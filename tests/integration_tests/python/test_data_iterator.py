"""TODO_doc."""

import pytest

from queens.main import run
from queens.utils.io_utils import load_result


def test_branin_data_iterator(inputdir, tmp_path, mocker, ref_result_iterator):
    """Test case for data iterator."""
    output = {}
    output['result'] = ref_result_iterator

    samples = ref_result_iterator

    mocker.patch(
        'queens.iterators.data_iterator.DataIterator.read_pickle_file',
        return_value=[samples, output],
    )

    run(inputdir / 'data_iterator_branin.yml', tmp_path)
    results = load_result(tmp_path / 'xxx.pickle')
    assert results["mean"] == pytest.approx(1.3273452195599997)
    assert results["var"] == pytest.approx(44.82468751096612)
