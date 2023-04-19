"""TODO_doc."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import seaborn as sns

cycle_colors = sns.color_palette()
style.use('seaborn')

"""TODO_doc: This is currently not in the documentation.

A module that provides utilities and a class for visualization in
sensitivity analysis (SA).

It is designed such that the SAVisualization class needs only to be initialized one
and can then be accessed and modified in the entire project.

In this context "this" is a pointer to the module object instance itself and can be compared to the
"self" keyword in classes.

Attributes:
    sa_visualization_instance (obj): Instance of the SAVisualization class
"""

this = sys.modules[__name__]
this.sa_visualization_instance = None


def from_config_create(plotting_options):
    """TODO_doc: add a one-line explanation.

    Module function that calls the class function *from_config_create* and
    creates instance of the SAVisualization class from the problem description.

    Args:
        plotting_options (dict): Dictionary containing the plotting options
    """
    this.sa_visualization_instance = SAVisualization.from_config_create(plotting_options)


def convert_to_dict(values):
    """Convert values to dictionary with plot keys.

    Args:
        values: TODO_doc

    Returns:
        plot_dict (dict): Data as dictionary with plot keys
    """
    plot_keys = ["bar", "scatter"]
    plot_dict = dict(zip(plot_keys, values))

    return plot_dict


def convert_to_pandas(results):
    """Convert results to pandas DataFrame.

    Args:
        results (dict): Data as dictionary

    Returns:
        output (DataFrame): Data as pandas DataFrame with parameter names as index
    """
    output = pd.DataFrame.from_dict(results["sensitivity_indices"])
    output = output.set_index('names')
    return output


def annotate_points(data):
    """Annotate points in scatter plot with parameter names.

    Args:
         data (DataFrame): Data to be annotated
    """
    for parameter in data.index.values:
        plt.annotate(
            parameter,
            (data.loc[parameter, 'mu_star'], data.loc[parameter, 'sigma']),
            textcoords="offset points",
            xytext=(0, 5),
            ha='center',
            fontsize=8,
        )


class SAVisualization(object):
    """TODO_doc: add a one-line explanation.

    Visualization class for sensitivity analysis that contains several
    plotting, storing and visualization methods that can be used anywhere in
    QUEENS.

    Attributes:
       saving_paths (dict): Dict of paths where to save the plots.
       should_be_saved (dict): Dict of booleans to save plots or not.
       should_be_displayed (dict): Dict of booleans for determining whether individual plots
                                   should be displayed or not.
       figures: TODO_doc

    Returns:
        SAVisualization (obj): Instance of the SAVisualization Class
    """

    # some overall class states
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams.update({'font.size': 28})

    def __init__(self, saving_paths, save_plot, display_plot):
        """TODO_doc.

        Args:
            saving_paths: TODO_doc
            save_plot: TODO_doc
            display_plot: TODO_doc
        """
        self.saving_paths = saving_paths
        self.should_be_saved = save_plot
        self.should_be_displayed = display_plot
        self.figures = {}

    @classmethod
    def from_config_create(cls, plotting_options):
        """Create the SAVisualization object from the problem description.

        Args:
            plotting_options (dict): Dictionary containing the plotting options

        Returns:
            Instance of SAVisualization (obj)
        """
        paths = [
            Path(plotting_options.get("plotting_dir"), name)
            for name in plotting_options["plot_names"]
        ]
        saving_paths = convert_to_dict(paths)

        save_booleans = plotting_options.get("save_bool")
        save_plot = convert_to_dict(save_booleans)

        plot_booleans = plotting_options.get("plot_booleans")
        display_plot = convert_to_dict(plot_booleans)

        return cls(saving_paths, save_plot, display_plot)

    def plot(self, results):
        """Call plotting methods for sensitivity analysis.

        Args:
            results (dict): Dictionary containing results to plot

        Returns:
            Plots of sensitivity indices
        """
        self.plot_si_bar(results)
        self.plot_si_scatter(results)

        # show all result plots in the end
        if any(self.should_be_displayed.values()) is True:
            self._display_plots()

    def plot_si_bar(self, results):
        """Plot the sensitivity indices as bar plot with error bars.

        Args:
            results (dict): Dictionary containing results to plot

        Returns:
            Plot of sensitivity indices as bar plot
        """
        if self.should_be_saved['bar'] or self.should_be_displayed['bar']:
            sensitivity_indices = convert_to_pandas(results)

            ax = sensitivity_indices.plot.bar(y='mu_star', yerr='sigma')

            ax.set_xlabel('Factors')
            ax.set_ylabel(r'$\mu^*_i$ and $\sigma_i$ (confidence intervals)')
            ax.set_title('Elementary Effects Analysis')

            ax.yaxis.grid(True)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            self._save_plot('bar')
            self.figures['bar'] = plt.gcf()

    def plot_si_scatter(self, results):
        """TODO_doc: add a one-line explanation.

        Plot the sensitivity indices as scatter plot of *sigma* over
        *mu_star*.

        Args:
            results (dict): Dictionary containing results to plot

        Returns:
            Plot of sensitivity indices as scatter plot
        """
        if self.should_be_saved['scatter'] or self.should_be_displayed['scatter']:
            sensitivity_indices = convert_to_pandas(results)

            ax = sensitivity_indices.plot.scatter(x='mu_star', y='sigma')
            annotate_points(sensitivity_indices)

            ax.set_xlabel(r'$\mu^*_i$')
            ax.set_ylabel(r'$\sigma_i$')
            ax.set_title('Elementary Effects Analysis')

            ax.yaxis.grid(True)
            plt.tight_layout()

            self._save_plot('scatter')
            self.figures['scatter'] = plt.gcf()

    def _display_plots(self):
        """Show plots according to input plot_booleans.

        Return:
            Displays plots.
        """
        for plot_key, current_figure in self.figures.items():
            if self.should_be_displayed[plot_key] is not True:
                plt.close(current_figure)

        plt.show()

    def _save_plot(self, key):
        """Save the plot to specified path.

        Args:
            key (str): key of current plot

        Returns:
            Saved plot.
        """
        if self.should_be_saved[key] is True:
            plt.savefig(self.saving_paths[key], dpi=300)
