"""TODO_doc."""

import pytest

from queens.visualization.sa_visualization import SAVisualization


@pytest.fixture(name="dummy_vis")
def fixture_dummy_vis(tmp_path):
    """Generate dummy instance of class SAVisualization."""
    paths = [
        tmp_path / name for name in ["test_sa_visualization_bar", "test_sa_visualization_scatter"]
    ]
    saving_paths = dict(zip(["bar", "scatter"], paths))
    save_booleans = {"bar": True, "scatter": True}
    plot_booleans = {"bar": True, "scatter": True}
    sa_vis = SAVisualization(saving_paths, save_booleans, plot_booleans)

    return sa_vis


def test_init(tmp_path, dummy_vis):
    """Test initialization of SAVisualization.

    Raises:
        AssertionError: If not correctly initialized
    """
    # expected attributes
    paths = [
        tmp_path / name for name in ["test_sa_visualization_bar", "test_sa_visualization_scatter"]
    ]
    saving_paths = dict(zip(["bar", "scatter"], paths))
    save_booleans = {"bar": True, "scatter": True}
    plot_booleans = {"bar": True, "scatter": True}

    assert dummy_vis.saving_paths == saving_paths
    assert dummy_vis.should_be_saved == save_booleans
    assert dummy_vis.should_be_displayed == plot_booleans


@pytest.fixture(name="dummy_sensitivity_indices")
def fixture_dummy_sensitivity_indices():
    """Generate dummy output data.

    Returns:
        results (dict): Contains dummy sensitivity indices (*names*, *mu*, *mu_star*,
        *sigma*, *mu_star_conf*)
    """
    results = {"sensitivity_indices": {}}
    results["sensitivity_indices"]["names"] = ["youngs1", "youngs2", "nue", "beta", "pressure"]
    results["sensitivity_indices"]["mu"] = [-0.24, -0.25, -1.42, -0.32, 0.78]
    results["sensitivity_indices"]["mu_star"] = [0.24, 0.25, 1.42, 0.32, 0.78]
    results["sensitivity_indices"]["sigma"] = [0.03, 0.10, 0.24, 0.01, 0.18]

    return results


def test_sa_visualization_bar(tmp_path, dummy_vis, dummy_sensitivity_indices):
    """Test bar plot generation and saving functionality.

    Test whether bar plot of sensitivity indices is plotting and saving the
    plot as a file.

    Raises:
        AssertionError: If no file was saved
    """
    dummy_vis.plot_si_bar(dummy_sensitivity_indices)

    path_output_image = tmp_path / "test_sa_visualization_bar.png"
    assert path_output_image.is_file()


def test_sa_visualization_scatter(tmp_path, dummy_vis, dummy_sensitivity_indices):
    """Test scatter plot generation and saving functionality.

    Test whether scatter plot of sensitivity indices is plotting and saving
    the plot as a file.

    Raises:
        AssertionError: If no file was saved
    """
    dummy_vis.plot_si_scatter(dummy_sensitivity_indices)

    path_output_image = tmp_path / "test_sa_visualization_scatter.png"
    assert path_output_image.is_file()
