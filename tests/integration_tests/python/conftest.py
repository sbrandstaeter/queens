"""Global fixtures and configurations for integration tests."""

import numpy as np
import pandas as pd
import pytest

from queens.example_simulator_functions.gaussian_logpdf import STANDARD_NORMAL, gaussian_1d_logpdf
from queens.example_simulator_functions.park91a import X3, X4, park91a_hifi_on_grid


@pytest.fixture(name="_create_experimental_data_gaussian")
def fixture_create_experimental_data_gaussian(tmp_path):
    """Create dummy Gaussian data."""
    # generate 10 samples from the same gaussian
    samples = STANDARD_NORMAL.draw(10).flatten()

    # evaluate the gaussian pdf for these 1000 samples
    pdf = []
    for sample in samples:
        pdf.append(gaussian_1d_logpdf(sample))

    pdf = np.array(pdf).flatten()

    # write the data to a csv file in tmp_path
    data_dict = {'y_obs': pdf}
    experimental_data_path = tmp_path / 'experimental_data.csv'
    dataframe = pd.DataFrame.from_dict(data_dict)
    dataframe.to_csv(experimental_data_path, index=False)


@pytest.fixture(name="_create_experimental_data_park91a_hifi_on_grid")
def fixture_create_experimental_data_park91a_hifi_on_grid(tmp_path):
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
        'x3': X3,
        'x4': X4,
        'y_obs': y_fake,
    }
    experimental_data_path = tmp_path / 'experimental_data.csv'
    dataframe = pd.DataFrame.from_dict(data_dict)
    dataframe.to_csv(experimental_data_path, index=False)


@pytest.fixture(name="_create_experimental_data_zero")
def fixture_create_experimental_data_zero(tmp_path):
    """Create 2 samples equal to zero."""
    samples = np.array([0, 0]).flatten()

    # write the data to a csv file in tmp_path
    data_dict = {'y_obs': samples}
    experimental_data_path = tmp_path / 'experimental_data.csv'
    dataframe = pd.DataFrame.from_dict(data_dict)
    dataframe.to_csv(experimental_data_path, index=False)
