import itertools
import os
import sys
import math

import matplotlib.pyplot as plt
from matplotlib import cm

from pqueens.visualization.custom_decision_boundary import DecisionBoundaryDisplay

"""
Provides a class for the neural iterator.
The NeuralIteratorVisualization class only needs to be initialized once and can then be accessed and modified in the entire project.

In this context "this" is a pointer to the module object instance itself and can be compared to the
"self" keyword in classes.

Attributes:
    neural_iterator_visualization_instance (obj): Instance of the NeuralIteratorVisualization class

Returns:
    None

"""

this = sys.modules[__name__]
this.neural_iterator_visualization_instance = None


def from_config_create(config, iterator_name='method'):
    """Module function that calls the class function `from_config_create` and
    creates instance of the NeuralIteratorVisualization class from the problem
    description.

    Args:
        config (dict): Dictionary created from the input file, containing the problem description
        iterator_name (str): Name of iterator to identify right section in options dict (optional)
    """
    this.neural_iterator_visualization_instance = NeuralIteratorVisualization.from_config_create(
        config, iterator_name
    )


class NeuralIteratorVisualization(object):
    """Visualization class for NeuralIterator that contains several plotting,
    storing and visualization methods that can be used anywhere in QUEENS.

    Attributes:
        saving_paths_list (list): List with saving_paths_list to save the plots.
        save_bools (list): List with booleans to save plots.
        plot_booleans (list): List of booleans for determining whether individual plots should be
                             plotted or not.
        var_names_list (list): List with variable names per parameter dimension

    Returns:
        NeuralIteratorVisualization (obj): Instance of the NeuralIteratorVisualization Class
    """

    # some overall class states
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams.update({'font.size': 22})

    def __init__(self, paths, save_bools, plot_booleans, var_names_list):
        self.saving_paths_list = paths
        self.save_bools = save_bools
        self.plot_booleans = plot_booleans
        self.var_names_list = var_names_list

    @classmethod
    def from_config_create(cls, config, iterator_name):
        """
        Create the visualization object from the problem description
        Args:
            config (dict): Dictionary containing the problem description
            iterator_name (str): Name of iterator to identify right section in options dict
                                 (optional)

        Returns:
            Instance of NeuralIteratorVisualization (obj)

        """
        method_options = config[iterator_name].get("method_options")

        plotting_options = method_options["result_description"].get("plotting_options")
        paths = [
            os.path.join(plotting_options.get("plotting_dir"), name)
            for name in plotting_options["plot_names"]
        ]
        save_bools = plotting_options.get("save_bool")
        plot_booleans = plotting_options.get("plot_booleans")

        # get the variable names
        random_variables = config[iterator_name]["parameters"].get("random_variables")
        var_names_list = []
        if random_variables is not None:
            for variable_name, _ in random_variables.items():
                var_names_list.append(variable_name)

        return cls(paths, save_bools, plot_booleans, var_names_list)

    def plot_decision_boundary(self, output, samples, num_params, clf):
        """Plot decision boundary of the trained classifier.

        If num_params is greater than 2, each combination of 2 parameters is plotted in a separate subplot.

        Args:
            output (dict):       Classification results obtained from simulation
            samples (np.array):     Array with sample points, size: (num_sample_points, num_params)
            num_params (int):    Number of parameters varied
            clf (Classifier):    utils.convergence_classifiers.Classifier instance for evaluation
        """
        if self.plot_booleans[0] is True or self.save_bools[0] is True:
            num_combinations, idx = _get_axes(num_params)
            fig, ax = plt.subplots(figsize=(10 * num_combinations, 10), ncols=num_combinations)
            for i, params in enumerate(idx):
                axis = ax[i] if clf.n_params > 2 else ax
                disp = DecisionBoundaryDisplay.from_estimator(
                    clf._clf,
                    samples[:, params],
                    n_params=clf.n_params,
                    params=params,
                    ax=axis,
                    response_method="predict",
                    cmap=cm.coolwarm,
                    alpha=0.8,
                    xlabel=self.var_names_list[params[0]],
                    ylabel=self.var_names_list[params[1]],
                )
                axis.scatter(samples[:, params[0]], samples[:, params[1]], c=output, cmap=cm.coolwarm, s=20,
                             edgecolors="k")
            _save_plot(self.save_bools[0], self.saving_paths_list[0])

        if self.plot_booleans[0] is True:
            plt.show()


def _get_axes(num_params):
    """
    Get all parameter combinations for plotting higher-dimensional decision boundaries.

    Args:
        num_params (int): Number of parameters varied

    Returns:
        number of combinations, list of parameter indexes
    """
    num_combinations = math.comb(num_params, 2)
    idx = itertools.combinations(range(num_params), 2)
    return num_combinations, idx


def _save_plot(save_bool, path):
    """Save the plot to specified path.

    Args:
        save_bool (bool): Flag to decide whether saving option is triggered.
        path (str): Path where to save the plot.

    Returns:
        Saved plot.
    """
    if save_bool is True:
        plt.savefig(path, dpi=300)
