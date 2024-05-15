"""TODO_doc."""
import pytest

from queens.global_settings import GlobalSettings
from queens.iterators.data_iterator import DataIterator
from queens.main import run_iterator
from queens.parameters.parameters import Parameters
from queens.utils.io_utils import load_result


def test_branin_data_iterator(tmp_path, mocker, ref_result_iterator, _initialize_global_settings):
    """Test case for data iterator."""
    # Global settings
    experiment_name = "branin_data_iterator"
    output_dir = tmp_path

    output = {}
    output['result'] = ref_result_iterator

    samples = ref_result_iterator

    mocker.patch(
        'queens.iterators.data_iterator.DataIterator.read_pickle_file',
        return_value=[samples, output],
    )

    with GlobalSettings(experiment_name=experiment_name, output_dir=output_dir, debug=False):
        # Parameters
        parameters = Parameters()

        # Setup QUEENS stuff
        iterator = DataIterator(
            path_to_data="/path_to_data/some_data.pickle",
            result_description={
                "write_results": True,
                "plot_results": False,
                "num_support_points": 5,
            },
            parameters=parameters,
            global_settings=_initialize_global_settings,
        )

        # Actual analysis
        run_iterator(
            iterator,
            global_settings=_initialize_global_settings,
        )
        # Load results
        result_file = output_dir / f"{experiment_name}.pickle"
        results = load_result(result_file)
    assert results["mean"] == pytest.approx(1.3273452195599997)
    assert results["var"] == pytest.approx(44.82468751096612)
