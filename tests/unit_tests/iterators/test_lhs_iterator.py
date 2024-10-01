"""Unit tests for LHS iterator."""

import numpy as np
import pytest

from queens.iterators.lhs_iterator import LHSIterator


@pytest.fixture(name="default_lhs_iterator")
def fixture_default_lhs_iterator(
    global_settings, default_simulation_model, default_parameters_mixed
):
    """Default latin hypercube sampling iterator."""
    default_simulation_model.driver.parameters = default_parameters_mixed

    # create LHS iterator
    # pylint: disable=duplicate-code
    my_iterator = LHSIterator(
        model=default_simulation_model,
        parameters=default_parameters_mixed,
        global_settings=global_settings,
        seed=42,
        num_samples=100,
        num_iterations=1,
        result_description=None,
        criterion="maximin",
    )
    return my_iterator


def test_correct_sampling(default_lhs_iterator):
    """Test if we get correct samples."""
    # np.set_printoptions(precision=10)
    default_lhs_iterator.pre_run()

    # check if mean and std match
    means_ref = np.array([-1.4546056001e-03, 5.4735307403e-03, 2.1664850171e00])

    np.testing.assert_allclose(
        np.mean(default_lhs_iterator.samples, axis=0), means_ref, 1e-09, 1e-09
    )

    std_ref = np.array([1.8157451781, 1.9914892803, 2.4282341125])
    np.testing.assert_allclose(np.std(default_lhs_iterator.samples, axis=0), std_ref, 1e-09, 1e-09)

    # check if samples are identical too
    ref_sample_first_row = np.array([-2.7374616292, -0.6146554017, 1.3925529817])

    np.testing.assert_allclose(
        default_lhs_iterator.samples[0, :], ref_sample_first_row, 1e-07, 1e-07
    )


def test_correct_results(default_lhs_iterator, ref_result_iterator):
    """Test if we get correct results."""
    default_lhs_iterator.pre_run()
    default_lhs_iterator.core_run()

    # np.set_printoptions(precision=10)

    # check if samples are identical too

    np.testing.assert_allclose(
        default_lhs_iterator.output["result"][0:10], ref_result_iterator, 1e-09, 1e-09
    )
