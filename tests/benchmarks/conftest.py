"""Configuration module for the benchmark tests."""

import pytest


@pytest.fixture(name="paths_dictionary")
def fixture_paths_dictionary(tmp_path, third_party_inputs, fourc_link_paths):
    """Create dictionary with directories for input file."""
    experimental_data_path = third_party_inputs / "csv" / "scatra_fourc"

    third_party_input_file_hf = third_party_inputs / "fourc" / "diffusion_coarse.dat"
    third_party_input_file_lf = third_party_inputs / "fourc" / "diffusion_very_coarse.dat"

    fourc_executable, post_ensight, _ = fourc_link_paths

    plot_dir = tmp_path

    paths_dictionary = {
        "experimental_data_path": experimental_data_path,
        "fourc_hf_input": third_party_input_file_hf,
        "fourc_lf_input": third_party_input_file_lf,
        "fourc-executable": fourc_executable,
        "post_ensight": post_ensight,
        "plot_dir": plot_dir,
    }
    return paths_dictionary
