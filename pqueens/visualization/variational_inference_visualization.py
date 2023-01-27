"""A module that provides utilities and a class for visualization in VI.

It is designed such that the VIVisualization class needs only to be
initialized once and can then be accessed and modified in the entire project.

In this context "this" is a pointer to the module object instance itself and can be compared to the
"self" keyword in classes.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

this = sys.modules[__name__]
this.vi_visualization_instance = None


def from_config_create(config):
    """Calls *from_config_create* and creates a singleton like object.

    Args:
        config (dict): Dictionary created from the input file, containing the problem description
    """
    this.vi_visualization_instance = VIVisualization.from_config_create(config)


class VIVisualization(object):
    """Visualization class for VI.

    Attributes:
       path (str): Paths to save the plots.
       save_bool (bool): Boolean to save plot.
       plot_boolean (bool): Boolean for determining whether should be plotted or not.
       axs_convergence_plots (matplotlib axes): Axes for the convergence plot.
       fig_convergence_plots (matplotlib figure): Figure for the convergence plot.
    """

    # some overall class states
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams.update({'font.size': 10})

    def __init__(self, path, save_bool, plot_boolean, axs_convergence_plots, fig_convergence_plots):
        """Initialize visualization object.

        Args:
            path (str): Paths to save the plots.
            save_bool (bool): Boolean to save plot.
            plot_boolean (bool): Boolean for determining whether should be plotted or not.
            axs_convergence_plots (matplotlib axes): Axes for the convergence plot
            fig_convergence_plots (matplotlib figure): Figure for the convergence plot
        """
        self.path = path
        self.save_bool = save_bool
        self.plot_boolean = plot_boolean
        self.axs_convergence_plots = axs_convergence_plots
        self.fig_convergence_plots = fig_convergence_plots

    @classmethod
    def from_config_create(cls, config):
        """Create the VIVisualization object from config.

        Args:
            config (dict): Dictionary containing the problem description

        Returns:
            Instance of VIVisualization (obj)
        """
        method_options = config["method"]
        plotting_options = method_options["result_description"].get("plotting_options")
        path = Path(plotting_options.get("plotting_dir"), plotting_options["plot_name"])
        save_bool = plotting_options.get("save_bool")
        plot_boolean = plotting_options.get("plot_boolean")
        axs_convergence_plots = None
        fig_convergence_plots = None
        return cls(path, save_bool, plot_boolean, axs_convergence_plots, fig_convergence_plots)

    def plot_convergence(self, iteration, variational_params_list, elbo):
        """Plots for VI over iterations.

        Consists of 3 subplots:
            1. ELBO
            2. Variational parameters
            3. Relative change in variational parameters

        Args:
            iteration (int): Current iteration
            variational_params_list (list): List of parameters from first to last iteration
            elbo (np.array): Row vector elbo values over iterations
        """
        if iteration > 1 and self.plot_boolean:
            iterations = np.arange(iteration)
            variational_params_array = np.array(variational_params_list)
            relative_change = np.abs(np.diff(variational_params_array, axis=0)) / np.maximum(
                np.abs(variational_params_array[:-1]), 1e-6
            )
            relative_change = np.mean(relative_change, axis=1)
            if self.fig_convergence_plots is None:
                self.fig_convergence_plots, self.axs_convergence_plots = plt.subplots(1, 3, num=2)
                self.fig_convergence_plots.set_size_inches(25, 8)
            self.axs_convergence_plots[0].clear()
            self.axs_convergence_plots[1].clear()
            self.axs_convergence_plots[0].plot(iterations, elbo, 'k-')
            self.axs_convergence_plots[1].plot(iterations, variational_params_array, '-')
            self.axs_convergence_plots[2].plot(iterations[1:], relative_change, 'k-')
            self.axs_convergence_plots[2].hlines(0.1, 0, iterations[-1], color='g')
            self.axs_convergence_plots[2].hlines(0.01, 0, iterations[-1], color='r')

            # ---- some further settings for the axes ---------------------------------------
            self.axs_convergence_plots[0].set_xlabel('iter.')
            self.axs_convergence_plots[0].set_ylabel('ELBO')
            self.axs_convergence_plots[0].grid(which='major', linestyle='-')
            self.axs_convergence_plots[0].grid(which='minor', linestyle='--', alpha=0.5)
            self.axs_convergence_plots[0].minorticks_on()

            self.axs_convergence_plots[1].set_xlabel('iter.')
            self.axs_convergence_plots[1].set_ylabel('Var. params.')
            self.axs_convergence_plots[1].grid(which='major', linestyle='-')
            self.axs_convergence_plots[1].grid(which='minor', linestyle='--', alpha=0.5)
            self.axs_convergence_plots[1].minorticks_on()

            self.axs_convergence_plots[2].set_xlabel('iter.')
            self.axs_convergence_plots[2].set_ylabel('Rel. change var. params.')
            self.axs_convergence_plots[2].set_yscale("log")
            self.axs_convergence_plots[2].grid(which='major', linestyle='-')
            self.axs_convergence_plots[2].grid(which='minor', linestyle='--', alpha=0.5)
            self.axs_convergence_plots[2].minorticks_on()
            plt.pause(0.0005)

    def save_plots(self):
        """Save the plot to specified path."""
        ###    Args:
        #           save_bool (bool): Flag to decide whether saving option is triggered
        #           path (str): Path where to save the plot

        if self.save_bool:
            self.fig_convergence_plots.savefig(self.path, dpi=300)
