import matplotlib.pyplot as plt
import math
import numpy as np
import os
import sys
from scipy.stats import norm
from scipy.stats import lognorm

"""
A module that provides utilities and a class for visualization in BMFMC-UQ
It is designed such that the BMFMCVisualization class needs only to be initialized one
and can then be accessed and modified in the entire project.

In this context "this" is a pointer to the module object instance itself and can be compared to the 
"self" keyword in classes.

Attributes:
    bmfmc_visualization_instance (obj): Instance of the BMFMCVisualization class

Returns:
    None
"""

this = sys.modules[__name__]
this.vi_visualization_instance = None


def from_config_create(config):
    """
    Module function that calls the class function `from_config_create` and creates instance of the
    BMFMCVisualization class from the problem description.

    Args:
        config (dict): Dictionary created from the input file, containing the problem description
    """
    this.vi_visualization_instance = VIVisualization.from_config_create(config)


class VIVisualization(object):
    """
    Visualization class for BMFMC-UQ that containts several plotting, storing and visualization
    methods that can be used anywhere in QUEENS.

    Attributes:
       paths (list): List with paths to save the plots.
       save_bools (list): List with booleans to save plots.
       animation_bool (bool): Flag for animation of 3D plots.
       predictive_var (bool): Flag for predictive variance plots.
       no_features_ref (bool): Flag for BMFMC-reference without informative features plot.
       plot_booleans (list): List of booleans for determining whether individual plots should be
                             plotted or not.

    Returns:
        BMFMCVisualization (obj): Instance of the BMFMCVisualization Class
    """

    # some overall class states
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams.update({'font.size': 10})

    def __init__(self, paths, save_bools, plot_booleans, MLE_comparison):
        self.paths = paths
        self.save_bools = save_bools
        self.plot_booleans = plot_booleans
        self.plot_list = []
        self.axs_1 = None
        self.fig_1 = None
        self.axs_2 = None
        self.fig_2 = None
        self.MLE_comparison = MLE_comparison
        self.iter = 0

    @classmethod
    def from_config_create(cls, config):
        """
        Create the BMFMC visualization object from the problem description
        Args:
            config (dict): Dictionary containing the problem description

        Returns:
            Instance of BMFMCVisualization (obj)

        """
        method_options = config["method"].get("method_options")
        plotting_options = method_options["result_description"].get("plotting_options")
        paths = [
            os.path.join(plotting_options.get("plotting_dir"), name)
            for name in plotting_options["plot_names"]
        ]
        save_bools = plotting_options.get("save_bool")
        plot_booleans = plotting_options.get("plot_booleans")
        MLE_comparison = plotting_options.get("MLE_comparison")
        return cls(paths, save_bools, plot_booleans, MLE_comparison)

    def plot_gaussian_pdfs_params(self, mean_params, var_params, transformation):
        """
        Plot the output distributions of HF output prediction, LF output, Monte-Carlo reference,
        posterior variance or BMFMC reference with no features, depending on settings in input file.
        Animate and save plots dependent on problem description.

        Args:
            output (dict): dictionary containing key-value paris to plot

        Returns:
            Plots of model output distribution

        """
        var_params = np.exp(2.0 * var_params)
        if self.plot_booleans[0]:
            if self.fig_1 is None:
                num = mean_params.size
                if num <= 5:
                    columns = num
                    rows = 1
                else:
                    columns = 5
                    rows = math.ceil(num / 5)
                self.fig_1, self.axs_1 = plt.subplots(nrows=rows, ncols=columns)
            for num, (ax, mu, var) in enumerate(zip(self.axs_1.flatten(), mean_params, var_params)):
                if transformation is None:
                    x_min = -1  # mu - 3 * np.sqrt(var)
                    x_max = 2  # mu + 3 * np.sqrt(var)
                    x_vec = np.linspace(x_min, x_max, 500)
                    x_vec_pdf = x_vec
                elif transformation == 'exp':
                    # transform to gaussian and set range
                    distr = lognorm(np.sqrt(var), loc=mu)
                    x_min = np.exp(distr.ppf(0.0)) * 0.5
                    x_max = np.exp(distr.ppf(0.9))
                    x_vec = np.linspace(x_min, x_max, 500)
                    x_vec_pdf = np.log(x_vec)
                else:
                    raise ValueError(f'Transformation {transformation} unknown! Abort ...')
                ax.clear()
                y_vec = norm.pdf(x_vec_pdf, scale=np.sqrt(var), loc=mu)
                ax.plot(
                    x_vec, y_vec, color='black', linewidth=1,
                )
                if self.MLE_comparison is not None:
                    ax.vlines(self.MLE_comparison[num], 0, 1.2 * np.max(y_vec), 'r')
                ax.set_title(f'$x_{num}$')

                # ---- some further settings for the axes ---------------------------------------
                ax.set_xlabel(f'$x_{num}$')
                ax.set_ylabel(f'$p(x_{num})$')
                ax.grid(which='major', linestyle='-')
                ax.grid(which='minor', linestyle='--', alpha=0.5)
                ax.minorticks_on()
                plt.pause(0.0005)
                self.iter += 1

            self.fig_1.set_size_inches(20, 8)

    def plot_convergence(self, iterations, variational_params_array, elbo, relative_change):
        if self.plot_booleans[1]:
            if self.fig_2 is None:
                self.fig_2, self.axs_2 = plt.subplots(1, 3, num=2)
            self.axs_2[0].clear()
            self.axs_2[1].clear()
            self.axs_2[0].plot(iterations, elbo[3:], 'k-')
            self.axs_2[1].plot(iterations, variational_params_array.T, '-')
            self.axs_2[2].plot(iterations[0:], np.mean(np.abs(relative_change), axis=1), 'k-')
            self.axs_2[2].hlines(0.1, 0, iterations[-1], color='g')
            self.axs_2[2].hlines(0.01, 0, iterations[-1], color='r')

            # ---- some further settings for the axes ---------------------------------------
            self.axs_2[0].set_xlabel('iter.')
            self.axs_2[0].set_ylabel('ELBO')
            self.axs_2[0].set_ylim([-1000.0, 200])
            self.axs_2[0].grid(which='major', linestyle='-')
            self.axs_2[0].grid(which='minor', linestyle='--', alpha=0.5)
            self.axs_2[0].minorticks_on()

            self.axs_2[1].set_xlabel('iter.')
            self.axs_2[1].set_ylabel('Var. params.')
            self.axs_2[1].grid(which='major', linestyle='-')
            self.axs_2[1].grid(which='minor', linestyle='--', alpha=0.5)
            self.axs_2[1].minorticks_on()

            self.axs_2[2].set_xlabel('iter.')
            self.axs_2[2].set_ylabel('Rel. change var. params.')
            self.axs_2[2].set_yscale("log")
            self.axs_2[2].grid(which='major', linestyle='-')
            self.axs_2[2].grid(which='minor', linestyle='--', alpha=0.5)
            self.axs_2[2].minorticks_on()

            self.fig_2.set_size_inches(25, 8)
            plt.pause(0.0005)

    def save_plots(self):
        """
        Save the plot to specified path.

        Args:
            save_bool (bool): Flag to decide whether saving option is triggered.
            path (str): Path where to save the plot.

        Returns:
            Saved plot.

        """
        if self.save_bools[0]:
            self.fig_1.savefig(self.paths[0], dpi=300)
        if self.save_bools[1]:
            self.fig_2.savefig(self.paths[1], dpi=300)
