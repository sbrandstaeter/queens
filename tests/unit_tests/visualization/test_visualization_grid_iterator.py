"""TODO_doc."""

import re

import numpy as np
import pytest

import queens.visualization.grid_iterator_visualization as qvis
from queens.visualization.grid_iterator_visualization import GridIteratorVisualization


class TestVisualizationGridIterator:
    """Collection of tests for visualisation features for the grid iterator."""

    @pytest.fixture(autouse=True)
    def dummy_vis(self, tmp_path):
        """A mock implementation of the qvis singleton for grid iterator.

        It will be used by all test in the test class
        """
        paths = [tmp_path / 'myplot.png']
        save_bools = [True]
        plot_booleans = [False]
        scale_types_list = ["lin", "lin"]
        var_names_list = ["x1", "x2"]
        grid_vis = GridIteratorVisualization(
            paths, save_bools, plot_booleans, scale_types_list, var_names_list
        )
        qvis.grid_iterator_visualization_instance = grid_vis

    # ------------------------------- actual unit_tests --------------------------------------------
    def test_init(self, tmp_path):
        """TODO_doc."""
        # expected attributes
        paths = [tmp_path / 'myplot.png']
        save_bools = [True]
        plot_booleans = [False]
        scale_types_list = ["lin", "lin"]
        var_names_list = ["x1", "x2"]

        # instantiated class
        grid_vis = GridIteratorVisualization(
            paths, save_bools, plot_booleans, scale_types_list, var_names_list
        )

        # tests / asserts
        assert grid_vis.saving_paths_list == paths
        assert grid_vis.save_bools == save_bools
        assert grid_vis.plot_booleans == plot_booleans
        assert grid_vis.scale_types_list == scale_types_list
        assert grid_vis.var_names_list == var_names_list

    def test_plot_qoi_grid(self, tmp_path):
        """TODO_doc."""
        # set arguments
        output = {"result": np.array([1.0, 2.0, 3.0, 4.0])}
        samples = np.array([[1.0, 1.0], [2.0, 2.0], [1.0, 2.0], [2.0, 1.0]])
        num_params = 2
        n_grid_p = [2, 2]
        qvis.grid_iterator_visualization_instance.plot_QoI_grid(
            output, samples, num_params, n_grid_p
        )
        filepath = tmp_path / 'myplot.png'
        assert filepath.is_file()

    def test_get_plotter_one(self):
        """TODO_doc."""
        num_params = 1
        plotter = qvis.grid_iterator_visualization_instance.get_plotter(num_params)
        expected_str = re.split("[\\s\\.]", str(plotter))[3]
        assert "plot_one_d" == expected_str

    def test_get_plotter_two(self):
        """TODO_doc."""
        num_params = 2
        plotter = qvis.grid_iterator_visualization_instance.get_plotter(num_params)
        expected_str = re.split("[\\s\\.]", str(plotter))[3]
        assert "plot_two_d" == expected_str

    def test_higher_d(self):
        """TODO_doc."""
        num_params = 3
        with pytest.raises(NotImplementedError) as not_implemented_error:
            qvis.grid_iterator_visualization_instance.get_plotter(num_params)
        assert str(not_implemented_error.value) == 'Grid plot only possible up to 2 parameters'

    def test_plot_one_d(self):
        """TODO_doc."""
        output = {'result': np.array([0.0, 1.0])}
        samples = np.array([0.0, 1.0])
        dummy = 1.0
        qvis.grid_iterator_visualization_instance.plot_one_d(output, samples, dummy)

    def test_plot_two_d(self):
        """TODO_doc."""
        samples = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        output = {'result': np.array([0.0, 1.0, 0.0, 1.0])}
        n_grid_p = [2, 2]
        qvis.grid_iterator_visualization_instance.plot_two_d(output, samples, n_grid_p)
