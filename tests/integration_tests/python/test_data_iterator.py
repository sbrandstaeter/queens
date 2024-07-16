"""TODO_doc."""

import pytest

from queens.iterators.data_iterator import DataIterator
from queens.main import run_iterator
from queens.parameters.parameters import Parameters
from queens.utils.io_utils import load_result


def test_branin_data_iterator(mocker, ref_result_iterator, global_settings):
    """Test case for data iterator."""
    output = {}
    output["result"] = ref_result_iterator

    samples = ref_result_iterator

    mocker.patch(
        "queens.iterators.data_iterator.DataIterator.read_pickle_file",
        return_value=[samples, output],
    )

    parameters = Parameters()

    # Setup iterator
    iterator = DataIterator(
        path_to_data="/path_to_data/some_data.pickle",
        result_description={
            "write_results": True,
            "plot_results": False,
            "num_support_points": 5,
        },
        parameters=parameters,
        global_settings=global_settings,
    )

    # Actual analysis
    run_iterator(iterator, global_settings=global_settings)
    # Load results
    results = load_result(global_settings.result_file(".pickle"))

    assert results["mean"] == pytest.approx(1.3273452195599997)
    assert results["var"] == pytest.approx(44.82468751096612)
