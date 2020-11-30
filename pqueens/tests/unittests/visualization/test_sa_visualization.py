import pytest
import os
from pqueens.visualization.sa_visualization import SAVisualization


@pytest.fixture()
def dummy_vis(tmpdir):
    """
    Generate dummy instance of class SAVisualization

    Args:
        tmpdir (str): Temporary directory in which the tests are run

    Returns:
        sa_vis (SAVisualization object): Instance of class SAVisualization
    """
    paths = [
        os.path.join(tmpdir, name)
        for name in ["test_sa_visualization_bar", "test_sa_visualization_scatter"]
    ]
    saving_paths = dict(zip(["bar", "scatter"], paths))
    save_booleans = {'bar': True, 'scatter': True}
    plot_booleans = {'bar': True, 'scatter': True}
    sa_vis = SAVisualization(saving_paths, save_booleans, plot_booleans)

    return sa_vis


def test_init(tmpdir, dummy_vis):
    """
    Test initialization of SAVisualization

    Raises:
        AssertionError: If not correctly initialized.
    """
    # expected attributes
    paths = [
        os.path.join(tmpdir, name)
        for name in ["test_sa_visualization_bar", "test_sa_visualization_scatter"]
    ]
    saving_paths = dict(zip(["bar", "scatter"], paths))
    save_booleans = {'bar': True, 'scatter': True}
    plot_booleans = {'bar': True, 'scatter': True}

    assert dummy_vis.saving_paths == saving_paths
    assert dummy_vis.should_be_saved == save_booleans
    assert dummy_vis.should_be_displayed == plot_booleans


@pytest.fixture()
def dummy_sensitivity_indices():
    """
    Generate dummy output data

    Returns:
        results (dict): Contains dummy sensitivity indices (names, mu, mu_star, sigma, mu_star_conf)
    """
    results = {"sensitivity_indices": {}}
    results["sensitivity_indices"]["names"] = ['youngs1', 'youngs2', 'nue', 'beta', 'pressure']
    results["sensitivity_indices"]["mu"] = [-0.24, -0.25, -1.42, -0.32, 0.78]
    results["sensitivity_indices"]["mu_star"] = [0.24, 0.25, 1.42, 0.32, 0.78]
    results["sensitivity_indices"]["sigma"] = [0.03, 0.10, 0.24, 0.01, 0.18]

    return results


def test_sa_visualization_bar(tmpdir, dummy_vis, dummy_sensitivity_indices):
    """
    Test whether bar plot of sensitivity indices is plotting and saving the plot as a file

    Raises:
        AssertionError: If no file was saved.
    """
    dummy_vis.plot_si_bar(dummy_sensitivity_indices)

    path_output_image = os.path.join(tmpdir, "test_sa_visualization_bar.png")
    assert os.path.isfile(path_output_image)


def test_sa_visualization_scatter(tmpdir, dummy_vis, dummy_sensitivity_indices):
    """
    Test whether scatter plot of sensitivity indices is plotting and saving the plot as a file

    Raises:
        AssertionError: If no file was saved.
    """

    dummy_vis.plot_si_scatter(dummy_sensitivity_indices)

    path_output_image = os.path.join(tmpdir, "test_sa_visualization_scatter.png")
    assert os.path.isfile(path_output_image)
