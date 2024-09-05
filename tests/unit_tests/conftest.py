"""Fixtures needed across unit_tests."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from queens.iterators.bmfia_iterator import BMFIAIterator
from queens.models.simulation_model import SimulationModel


@pytest.fixture(name="dummy_simulation_model")
def fixture_dummy_simulation_model():
    """Dummy model."""
    model = SimulationModel(scheduler=Mock(), driver=Mock())
    return model


@pytest.fixture(name="get_patched_bmfia_iterator")
def fixture_get_patched_bmfia_iterator(global_settings):
    """Function that returns a dummy BMFIA iterator for testing."""

    def get_patched_bmfia_iterator(parameters, hf_model, lf_model):
        x_train = np.array([[1, 2], [3, 4]])
        features_config = "no_features"
        x_cols = None
        num_features = None
        coord_cols = None

        with patch.object(BMFIAIterator, "calculate_initial_x_train", lambda *args: x_train):
            iterator = BMFIAIterator(
                parameters=parameters,
                global_settings=global_settings,
                features_config=features_config,
                hf_model=hf_model,
                lf_model=lf_model,
                initial_design={},
                X_cols=x_cols,
                num_features=num_features,
                coord_cols=coord_cols,
            )

        iterator.Y_LF_train = np.array([[2], [3]])
        iterator.Y_HF_train = np.array([[2.2], [3.3]])
        iterator.Z_train = np.array([[4], [5]])
        iterator.coords_experimental_data = np.array([[1, 2], [3, 4]])
        iterator.time_vec = np.array([1, 3])
        iterator.y_obs_vec = np.array([[2.1], [3.1]])

        return iterator

    return get_patched_bmfia_iterator


@pytest.fixture(name="result_description")
def fixture_result_description():
    """A dummy result description."""
    description = {"write_results": True}
    return description
