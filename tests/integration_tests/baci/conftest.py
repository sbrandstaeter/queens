"""Pytest configuration for BACI integration tests."""

import numpy as np
import pandas as pd
import pytest

from queens.example_simulator_functions.park91a import park91a_hifi_on_grid, x3_vec, x4_vec


@pytest.fixture(name="setup_symbolic_links_baci", autouse=True)
def fixture_setup_symbolic_links_baci(baci_link_paths, baci_source_paths_for_gitlab_runner):
    """Set-up of BACI symbolic links.

    Args:
        baci_link_paths (Path): destination for symbolic links to executables
        baci_source_paths_for_gitlab_runner (Path): Several paths that are needed to build symbolic
                                                links to executables
    """
    (
        dst_baci_release,
        dst_post_ensight,
        dst_post_processor,
    ) = baci_link_paths

    (
        src_baci_release,
        src_post_ensight,
        src_post_processor,
    ) = baci_source_paths_for_gitlab_runner
    # check if symbolic links are existent
    try:
        # create link to default baci-release location if no link is available
        if not dst_baci_release.is_symlink():
            if not src_baci_release.is_file():
                raise FileNotFoundError(
                    f'Failed to create link to default baci-release location.\n'
                    f'No baci-release found under default location:\n'
                    f'\t{src_baci_release}\n'
                )
            dst_baci_release.symlink_to(src_baci_release)
        # create link to default post_ensight location if no link is available
        if not dst_post_ensight.is_symlink():
            if not src_post_ensight.is_file():
                raise FileNotFoundError(
                    f'Failed to create link to default post_ensight location.\n'
                    f'No post_ensight found under default location:\n'
                    f'\t{src_post_ensight}\n'
                )
            dst_post_ensight.symlink_to(src_post_ensight)
        # create link to default post_processor location if no link is available
        if not dst_post_processor.is_symlink():
            if not src_post_processor.is_file():
                raise FileNotFoundError(
                    f'Failed to create link to default post_processor location.\n'
                    f'No post_processor found under default location:\n'
                    f'\t{src_post_processor}\n'
                )
            dst_post_processor.symlink_to(src_post_processor)

        # check if existing link to baci-release works and points to a valid file
        if not dst_baci_release.resolve().exists():
            raise FileNotFoundError(
                f'The following link seems to be dead: {dst_baci_release}\n'
                f'It points to (non-existing): {dst_baci_release.resolve()}\n'
            )
        # check if existing link to post_ensight works and points to a valid file
        if not dst_post_ensight.resolve().exists():
            raise FileNotFoundError(
                f'The following link seems to be dead: {dst_post_ensight}\n'
                f'It points to: {dst_post_ensight.resolve()}\n'
            )
        # check if existing link to post_processor works and points to a valid file
        if not dst_post_processor.resolve().exists():
            raise FileNotFoundError(
                f'The following link seems to be dead: {dst_post_processor}\n'
                f'It points to: {dst_post_processor.resolve()}\n'
            )
    except FileNotFoundError as error:
        raise FileNotFoundError(
            'Please make sure to make the missing executable available under the given '
            'path OR\n'
            'make sure the symbolic link under the config directory points to an '
            'existing file! \n'
            'You can create the necessary symbolic links on Linux via:\n'
            '-------------------------------------------------------------------------\n'
            'ln -s <path/to/baci-release> <QUEENS_BaseDir>/config/baci-release\n'
            'ln -s <path/to/post_ensight> <QUEENS_BaseDir>/config/post_ensight\n'
            'ln -s <path/to/post_processor> <QUEENS_BaseDir>/config/post_processor\n'
            '-------------------------------------------------------------------------\n'
            '...and similar for the other links.'
        ) from error


@pytest.fixture(name="create_experimental_data_park91a_hifi_on_grid")
def fixture_create_experimental_data_park91a_hifi_on_grid(tmp_path):
    """Create experimental data."""
    # Fix random seed
    np.random.seed(seed=1)

    # True input values
    x1 = 0.5  # pylint: disable=invalid-name
    x2 = 0.2  # pylint: disable=invalid-name

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
    experimental_data_path = tmp_path / 'experimental_data.csv'
    dataframe = pd.DataFrame.from_dict(data_dict)
    dataframe.to_csv(experimental_data_path, index=False)
