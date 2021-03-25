import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
import matplotlib.ticker as mticker
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import os
import sys

"""
A module that provides utilities and a class for visualization in the grid iterator.
It is designed such that the GridIteratorVisualization class needs only to be initialized one
and can then be accessed and modified in the entire project.

In this context "this" is a pointer to the module object instance itself and can be compared to the 
"self" keyword in classes.

Attributes:
    grid_iterator_visualization_instance (obj): Instance of the GridIteratorVisualization class
    
Returns:
    None
    
"""

this = sys.modules[__name__]
this.grid_iterator_visualization_instance = None


def from_config_create(config, iterator_name=None):
    """
    Module function that calls the class function `from_config_create` and creates instance of the
    GridIteratorVisualization class from the problem description.

    Args:
        config (dict): Dictionary created from the input file, containing the problem description
        iterator_name (str): Name of iterator to identify right section in options dict (optional)
    """
    this.grid_iterator_visualization_instance = GridIteratorVisualization.from_config_create(
        config, iterator_name
    )


def _log_tick_formatter(val):
    """
    Tick formatter for 10-logarithmic scaling

    Args:
        val (np.array): Values of grid variable

    Returns:
        Formatted tick values

    """
    return "{:.2e}".format(10 ** val)


def _ln_tick_formatter(val):
    """
    Tick formatter for natural logarithmic scaling

    Args:
        val (np.array): Values of grid variable

    Returns:
        Formatted tick values

    """

    return "{:.2e}".format(np.e ** val)


def _linear_tick_formatter(val):
    """
    Tick formatter for linear scaling

    Args:
        val (np.array): Values of grid variable

    Returns:
        Formatted tick values

    """

    return "{:.2e}".format(val)


class GridIteratorVisualization(object):
    """
    Visualization class for GridIterator that contains several plotting, storing and visualization
    methods that can be used anywhere in QUEENS.

    Attributes:
        saving_paths_list (list): List with saving_paths_list to save the plots.
        save_bools (list): List with booleans to save plots.
        plot_booleans (list): List of booleans for determining whether individual plots should be
                             plotted or not.
        scale_types_list (list): List scaling types for each grid variable
        var_names_list (list): List with variable names per grid dimension

    Returns:
        GridIteratorVisualization (obj): Instance of the GridIteratorVisualization Class
    """

    # some overall class states
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams.update({'font.size': 22})

    def __init__(self, paths, save_bools, plot_booleans, scale_types_list, var_names_list):
        self.saving_paths_list = paths
        self.save_bools = save_bools
        self.plot_booleans = plot_booleans
        self.scale_types_list = scale_types_list
        self.var_names_list = var_names_list

    @classmethod
    def from_config_create(cls, config, iterator_name=None):
        """
        Create the grid visualization object from the problem description
        Args:
            config (dict): Dictionary containing the problem description
            iterator_name (str): Name of iterator to identify right section in options dict
                                 (optional)

        Returns:
            Instance of GridIteratorVisualization (obj)

        """
        if iterator_name is None:
            method_options = config["method"].get("method_options")
        else:
            method_options = config[iterator_name].get("method_options")

        plotting_options = method_options["result_description"].get("plotting_options")
        paths = [
            os.path.join(plotting_options.get("plotting_dir"), name)
            for name in plotting_options["plot_names"]
        ]
        save_bools = plotting_options.get("save_bool")
        plot_booleans = plotting_options.get("plot_booleans")

        # get the variable names and the grid design
        grid_design = method_options.get('grid_design')
        var_names_list = []
        scale_types_list = []
        if grid_design is not None:
            for variable_name, grid_opt in grid_design.items():
                var_names_list.append(variable_name)
                scale_types_list.append(grid_opt.get('axis_type'))

        return cls(paths, save_bools, plot_booleans, scale_types_list, var_names_list)

    def plot_QoI_grid(self, output, samples, num_params, n_grid_p):
        """
        Plot Quantity of Interest over grid (so far support up to 2D grid)

        args:
            output (dict):       QoI obtained from simulation
            samples (np.array):  Grid coordinates flattened 1D arrays as columns of 2D samples array
            num_params (int):                 Number of parameters varied
            n_grid_p (np.array): Array containing number of grid points for each parameter

        """
        if self.plot_booleans[0] is True or self.save_bools[0] is True:
            plotter = self._get_plotter(num_params)
            plotter(output, samples, n_grid_p)
            _save_plot(self.save_bools[0], self.saving_paths_list[0])

        if self.plot_booleans[0] is True:
            plt.show()

    def _get_plotter(self, num_params):
        """
        Get the correct plotting function depending on the dimension of the grid

        Args:
            num_params (int): Number of grid-dimensions

        Returns:
            Plotting function for corresponding dimension (obj)

        """
        if num_params == 1:
            return self._plot_one_d
        elif num_params == 2:
            return self._plot_two_d
        else:
            raise NotImplementedError('Grid plot only possible up to 2 parameters')

    def _plot_one_d(self, output, samples, n_grid_p):
        """
        Plotting method for one dimensional grid

        Args:
            output (np.array): Simulation output
            samples (np.array): Simulation input/samples/grid-points
            n_grid_p (np.array): Array containing number of grid points for each parameter

        Returns:
            None

        """
        fig, ax = plt.subplots()

        # get axes
        x = samples
        y = output['mean']
        min_y = min(output['mean'])
        max_y = max(output['mean'])
        min_x = min(samples)
        max_x = max(samples)

        # --------------------- plot QoI over samples ---------------------
        ax.set_xscale('log')
        ax.set_yscale('linear')
        ax.plot(x, y)

        # set major/minor ticks for log scale
        ax.minorticks_on()
        ax.grid()
        ax.set_xlabel(f'{self.var_names_list[0]} [{self.scale_types_list[0]}]')
        ax.set_ylabel('QoI')

        # adjust limits of axes
        ax.set(xlim=(min_x, max_x), ylim=(min_y, max_y))

    def _plot_two_d(self, output, samples, n_grid_p):
        """
        Plotting method for two dimensional grid

        Args:
            output (np.array): Simulation output
            samples (np.array): Simulation input/samples/grid-points
            n_grid_p (np.array): Array containing number of grid points for each parameter

        Returns:
            None

        """
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        # get axes
        x = samples[:, 0].reshape(n_grid_p[0], n_grid_p[1])
        y = samples[:, 1].reshape(n_grid_p[0], n_grid_p[1])
        z = output['mean'].reshape(n_grid_p[0], n_grid_p[1])
        min_z = min(output['mean'])
        max_z = max(output['mean'])

        # --------------------- plot QoI over samples ---------------------
        surf = ax.plot_surface(np.log10(x), y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        x_formatter = self._get_tick_formatter('x')
        y_formatter = self._get_tick_formatter('y')

        # scale axes with user defined tick formatter
        # TODO the formatter contains currently a bug
        ax.xaxis.set_major_locator(plt.MaxNLocator(n_grid_p[0]))
        ax.yaxis.set_major_locator(plt.MaxNLocator(n_grid_p[1]))

        # Customize the z axis.
        ax.set_zlim(min_z, max_z)
        ax.zaxis.set_major_locator(LinearLocator(5))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # font, axes label and tick size
        ax.tick_params(labelsize='xx-small')

        # axes labels
        ax.set_xlabel(f'{self.var_names_list[0]} [{self.scale_types_list[0]}]')
        ax.set_ylabel(f'{self.var_names_list[1]} [{self.scale_types_list[1]}]')
        ax.set_zlabel('QoI')

        # Add a color bar (optional)
        fig.colorbar(surf, shrink=0.5, aspect=5)

    def _get_tick_formatter(self, axis_str):
        """
        Depending on the scaling of the grid axis, return an appropriate formatter for the
        axes ticks

        Args:
            axis_str (str): Identifier for either "x" or "y" axis of the grid

        Returns:
            tick_formatter (obj): Tick-formatter object

        """
        if axis_str == 'x':
            idx = 0
        elif axis_str == 'y':
            idx = 1
        else:
            raise ValueError('Axis string is not defined!')

        if self.scale_types_list[idx] == 'log10':
            return _log_tick_formatter
        elif self.scale_types_list[idx] == 'logn':
            return _ln_tick_formatter
        elif self.scale_types_list[idx] == 'lin':
            return _linear_tick_formatter
        else:
            raise ValueError(
                f'Your axis scaling type {self.scale_types_list[idx]} is not a valid '
                f'option! Abort...'
            )


def _save_plot(save_bool, path):
    """
    Save the plot to specified path.

    Args:
        save_bool (bool): Flag to decide whether saving option is triggered.
        path (str): Path where to save the plot.

    Returns:
        Saved plot.

    """
    if save_bool is True:
        plt.savefig(path, dpi=300)
