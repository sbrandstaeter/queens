from typing import Tuple

import numpy as np
import pandas as pd

import pqueens.visualization.neural_iterator_visualization as qvis
from pqueens.models import from_config_create_model
from pqueens.utils.process_outputs import process_ouputs, write_results
from pqueens.utils.convergence_classifiers import Classifier, SvmClassifier, NNClassifier, GPActiveLearningClassifier, \
    DeepActiveClassifier

from .iterator import Iterator

"""Machine learning based iterator."""


class NeuralIterator(Iterator):
    """Machine learning based iterator for learning parameter combinations where the solver converges.

    It is possible to choose between various machine learning classifiers implemented in utils/convergence_classifiers.py.

    Attributes:
        model (model): Model to be evaluated by iterator
        result_description (dict):  Description of desired results
        parameters (dict) :    dictionary containing parameter information
        num_parameters (int)          :   number of parameters to be varied
        samples (np.array):   Array with all samples
        output (np.array):   Array with all model outputs
        num_sample_points (int):  number of total sample points
        num_active_sample_points (int):  number of points to query in active learning (only for active learning classifiers)
    """

    def __init__(
            self,
            model,
            result_description,
            global_settings,
            parameters,
            num_parameters,
    ):
        """Initialize grid iterator.

        Args:
            model (model): Model to be evaluated by iterator
            result_description (dict):  Description of desired results
            global_settings (dict, optional): Settings for the QUEENS run.
            parameters (dict):    dictionary containing parameter information
            num_parameters (int):   number of parameters to be varied
        """
        super().__init__(model, global_settings)
        self.parameters = parameters
        self.result_description = result_description
        self.samples = None
        self.output = None
        self.num_sample_points = None
        self.num_active_sample_points = None
        self.num_parameters = num_parameters

    @classmethod
    def from_config_create_iterator(cls, config, iterator_name, model=None):
        """Create grid iterator from problem description.

        Args:
            config (dict):       Dictionary with QUEENS problem description
            iterator_name (str): Name of iterator to identify right section
                                 in options dict (optional)
            model (model):       Model to use (optional)


        Returns:
            iterator (obj): GridIterator object
        """
        method_options = config[iterator_name]["method_options"]
        if model is None:
            model_name = method_options["model"]
            model = from_config_create_model(model_name, config)

        result_description = method_options.get("result_description", None)
        global_settings = config.get("global_settings", None)
        parameters = config["parameters"]["random_variables"]
        num_parameters = len(parameters)

        # ---------------------- CREATE VISUALIZATION BORG ----------------------------
        qvis.from_config_create(config, iterator_name=iterator_name)

        return cls(
            model, result_description, global_settings, parameters, num_parameters
        )

    def eval_model(self):
        """Evaluate the model."""
        return self.model.evaluate()

    def pre_run(self):
        """Generate samples based on description in parameters."""
        np.random.seed(0)

        # set up array with uniformly distributed sample points
        num_sample_points = self.parameters
        self.samples = np.empty([self.num_sample_points, self.num_parameters])

        for index, (parameter_name, parameter) in enumerate(self.parameters.items()):
            start_value = parameter["lower_bound"]
            stop_value = parameter["upper_bound"]
            self.samples[:, index] = np.random.uniform(low=start_value, high=stop_value, size=num_sample_points).astype(np.float32)

    def core_run(self):
        """Evaluate the samples on model."""
        self.model.update_model_from_sample_batch(self.samples)
        self.output = self.eval_model()

    def post_run(self):
        """Analyze the results."""
        if self.result_description is not None:
            results = process_ouputs(self.output, self.result_description, self.samples)
            if self.result_description["write_results"] is True:
                write_results(
                    results,
                    self.global_settings["output_dir"],
                    self.global_settings["experiment_name"],
                )

        # plot decision boundary for the trained classifier
        qvis.neural_iterator_visualization_instance.plot_decision_boundary(
            self.output,
            self.samples,
            self.num_parameters,
            self.num_sample_points,
        )
        # TODO: save checkpoint


def _undersample(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray]:
    """
    filter out non-converging points from the imbalanced dataset (much more points with value 0 than 1)

    Args:
        X: array of parameter points, size: (n_samples, n_params)
        y: vector of target values, size: (n_samples)

    Returns:
        refined array X which contains fewer samples of class 0
        corresponding refined target value vector y
    """
    df = pd.DataFrame((y, X[:, 0], X[:, 1])).T
    df_minor = df.loc[df[0] == 1]
    df_sampled = df.loc[df[0] == 0].sample(n=df_minor.shape[0], random_state=0)
    df = pd.concat([df_minor, df_sampled])
    yX = df.sample(frac=1).to_numpy(dtype=np.float32)
    return yX[:, 1:], yX[:, 0]
