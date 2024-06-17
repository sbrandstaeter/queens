"""Configuration module for the benchmark tests."""

import pytest


@pytest.fixture(name="dir_dict")
def fixture_dir_dict(tmp_path, third_party_inputs, baci_link_paths):
    """Create dictionary with directories for input file."""
    experimental_data_path = third_party_inputs / "csv" / "scatra_baci"

    third_party_input_file_hf = third_party_inputs / "baci" / "diffusion_coarse.dat"
    third_party_input_file_lf = third_party_inputs / "baci" / "diffusion_very_coarse.dat"

    baci_release, post_ensight, _ = baci_link_paths

    plot_dir = tmp_path

    dir_dict = {
        'experimental_data_path': experimental_data_path,
        'baci_hf_input': third_party_input_file_hf,
        'baci_lf_input': third_party_input_file_lf,
        'baci-release': baci_release,
        'post_ensight': post_ensight,
        'plot_dir': plot_dir,
    }
    return dir_dict
