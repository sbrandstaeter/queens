"""TODO_doc."""


import numpy as np
import pandas as pd
import pytest

from pqueens.tests.integration_tests.example_simulator_functions.park91a import (
    park91a_hifi_on_grid,
    x3_vec,
    x4_vec,
)


@pytest.fixture()
def create_experimental_data_park91a_hifi_on_grid(tmp_path):
    """Create experimental data."""
    # Fix random seed
    np.random.seed(seed=1)

    # True input values
    x1 = 0.5
    x2 = 0.2

    y_vec = park91a_hifi_on_grid(x1, x2)

    # Artificial noise
    sigma_n = 0.001
    noise_vec = np.random.normal(loc=0, scale=sigma_n, size=(y_vec.size,))

    # Inverse crime: Add artificial noise to model output for the true value
    y_fake = y_vec + noise_vec

    # write fake data to csv
    data_dict = {
        'x3': x3_vec,
        'x4': x4_vec,
        'y_obs': y_fake,
    }
    experimental_data_path = tmp_path.joinpath('experimental_data.csv')
    df = pd.DataFrame.from_dict(data_dict)
    df.to_csv(experimental_data_path, index=False)
