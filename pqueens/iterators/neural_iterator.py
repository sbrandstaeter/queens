from typing import Tuple

import numpy as np
import pandas as pd
from skactiveml.utils import MISSING_LABEL
from tqdm import tqdm

import pqueens.visualization.neural_iterator_visualization as qvis
from pqueens.models import from_config_create_model
from pqueens.utils.process_outputs import process_ouputs, write_results
from pqueens.utils.convergence_classifiers import Classifier, SvmClassifier, NNClassifier, GPActiveLearningClassifier, \
    DeepActiveClassifier

from .iterator import Iterator

"""Machine learning based iterator.

This iterator trains a machine learning classification algorithm to learn the area of convergence for a given solver 
by collecting samples, optionally using an active learning strategy from scikit-activeml.
"""


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
            num_sample_points,
            num_active_sample_points,
            classifier_name,
    ):
        """Initialize neural iterator.

        Args:
            model (model): Model to be evaluated by iterator
            result_description (dict):  Description of desired results
            global_settings (dict, optional): Settings for the QUEENS run.
            parameters (dict):    dictionary containing parameter information
            num_parameters (int):   number of parameters to be varied
            num_sample_points (int): number of total points to choose from for active pooling
            num_active_sample_points (int): number of sample points to query (number of calls to the model)
            classifier_name (str): classifier algorithm to be used
        """
        super().__init__(model, global_settings)
        self.parameters = parameters
        self.result_description = result_description
        self.samples = None
        self.output = None
        self.num_sample_points = None
        self.num_active_sample_points = None
        self.num_parameters = num_parameters
        self.num_sample_points = num_sample_points
        self.num_active_sample_points = num_active_sample_points

        classifier_dict = {
            'svm': SvmClassifier,
            'mlp': NNClassifier,
            'active_gp': GPActiveLearningClassifier,
            'active_mlp': DeepActiveClassifier,
        }
        clf = classifier_dict.get(classifier_name)
        self.classifier = clf(num_parameters)

    @classmethod
    def from_config_create_iterator(cls, config, iterator_name, model=None):
        """Create neural iterator from problem description.

        Args:
            config (dict):       Dictionary with QUEENS problem description
            iterator_name (str): Name of iterator to identify right section
                                 in options dict (optional)
            model (model):       Model to use (optional)

        Returns:
            iterator (obj): NeuralIterator object
        """
        method_options = config[iterator_name]["method_options"]
        if model is None:
            model_name = method_options["model"]
            model = from_config_create_model(model_name, config)

        result_description = method_options.get("result_description", None)
        global_settings = config.get("global_settings", None)
        parameters = config["parameters"]["random_variables"]
        num_parameters = len(parameters)
        classifier_name = config["parameters"]["classifier"]
        num_sample_points = method_options["num_sample_points"]
        num_active_sample_points = method_options.get("num_active_sample_points", None)

        # ---------------------- CREATE VISUALIZATION BORG ----------------------------
        qvis.from_config_create(config, iterator_name=iterator_name)

        return cls(
            model, result_description, global_settings, parameters, num_parameters, num_sample_points, num_active_sample_points, classifier_name
        )

    def pre_run(self):
        """Generate samples based on description in parameters."""
        np.random.seed(0)

        # set up array with uniformly distributed sample points
        self.samples = np.empty([self.num_sample_points, self.num_parameters])

        for index, (parameter_name, parameter) in enumerate(self.parameters.items()):
            start_value = parameter["lower_bound"]
            stop_value = parameter["upper_bound"]
            self.samples[:, index] = np.random.uniform(low=start_value, high=stop_value, size=self.num_sample_points).astype(np.float32)

    def core_run(self):
        """Evaluate the samples on model."""
        if self.num_active_sample_points is None:
            self.output = self.model.evaluate(self.samples)
            converged = (~np.isnan(self.output["mean"][:, 0])).astype(np.int64)
            self.classifier.train(self.samples, converged)

        else:
            # use active learning
            converged = np.full(shape=self.num_sample_points, fill_value=MISSING_LABEL)
            query_idx = self.classifier.train(self.samples, converged)
            outputs = []
            query_idxs = []
            for _ in tqdm(range(self.num_active_sample_points)):
                output = self.model.evaluate(self.samples[query_idx])
                converged[query_idx] = (~np.isnan(output["mean"][:, 0])).astype(np.int64)
                query_idx = self.classifier.train(self.samples, converged)
                outputs.append(output["mean"])
                query_idxs.append(query_idx.item())
            self.output = {"mean": np.vstack(outputs)}
            self.samples = self.samples[query_idxs, :]

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
            self.classifier._clf,
        )

        # save checkpoint
        self.classifier.save(self.global_settings["output_dir"], self.global_settings["experiment_name"])


def _undersample(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray]:
    """
    Filter out non-converging points from the imbalanced dataset (much more points with value 0 than 1).

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
