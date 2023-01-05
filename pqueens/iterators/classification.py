"""Binary classification iterator.

This iterator trains a classification algorithm based on a forward and
classification model.
"""

import logging

import numpy as np
from skactiveml.utils import MISSING_LABEL

from pqueens.iterators.iterator import Iterator
from pqueens.models import from_config_create_model
from pqueens.utils.ascii_art import print_classification
from pqueens.utils.classifier import from_config_create_classifier
from pqueens.utils.import_utils import get_module_attribute
from pqueens.utils.process_outputs import write_results
from pqueens.visualization.classification import ClassificationVisualization

_logger = logging.getLogger(__name__)


class ClassificationIterator(Iterator):
    """Iterator for machine leaning based classification.

    Attributes:
        model (model): Model to be evaluated by iterator
        result_description (dict):  Description of desired results
        global_settings (dict, optional): Settings for the QUEENS run.
        num_sample_points (int): number of total points
        num_model_calls (int): total number of model calls
        random_sampling_frequecy (int): in case of active sampling every iteration index that
                                        is a multiple of this number the samples are selected
                                        randomly
        classifier (obj): queens classifier object
        visaulization_obj (obj): object for visualization
        classification_function (fun): function that classifies the model output
        samples (np.array): samples on which the model was evaluated at
        classified_outputs (np.array): classified output of evaluated at samples
    """

    def __init__(
        self,
        model,
        result_description,
        global_settings,
        num_sample_points,
        num_model_calls,
        random_sampling_frequency,
        classifier,
        seed,
        visualization_obj,
        classification_function,
    ):
        """Initialize neural iterator.

        Args:
            model (model): Model to be evaluated by iterator
            result_description (dict):  Description of desired results
            global_settings (dict, optional): Settings for the QUEENS run.
            num_sample_points (int): number of total points
            num_model_calls (int): total number of model calls
            random_sampling_frequency (int): in case of active sampling every iteration index that
                                             is a multiple of this number the samples are selected
                                             randomly
            classifier (obj): queens classifier object
            visualization_obj (obj): object for visualization
            classification_function (fun): function that classifies the model output
            seed (int): random initial seed
        """
        if classifier.is_active and num_model_calls >= num_sample_points:
            raise ValueError("Number of sample points needs to be greater than num_model_calls.")
        super().__init__(model, global_settings)
        self.result_description = result_description
        self.samples = np.empty((0, self.parameters.num_parameters))
        self.classified_outputs = np.empty((0, 1))
        self.num_sample_points = num_sample_points
        self.num_model_calls = num_model_calls
        self.classifier = classifier
        self.seed = seed
        self.visualization_obj = visualization_obj
        self.random_sampling_frequency = random_sampling_frequency
        self.classification_function = classification_function

        np.random.seed(self.seed)

    @classmethod
    def from_config_create_iterator(cls, config, iterator_name, model=None):
        """Create neural iterator from problem description.

        Args:
            config (dict):       Dictionary with QUEENS problem description
            iterator_name (str): Name of iterator to identify right section
                                 in options dict (optional)
            model (model):       Model to use (optional)

        Returns:
            iterator (obj): ClassificationIterator object
        """
        method_options = config[iterator_name]["method_options"]
        if model is None:
            model_name = method_options["model"]
            model = from_config_create_model(model_name, config)

        result_description = method_options.get("result_description")
        global_settings = config.get("global_settings")
        classifier_name = method_options["classifier"]
        num_sample_points = method_options["num_sample_points"]
        num_model_calls = method_options["num_model_calls"]

        random_sampling_frequency = method_options.get("random_sampling_frequency", 0)
        seed = method_options.get("seed", 42)

        if visualization_obj := method_options["result_description"].get("plotting_options"):
            visualization_obj = ClassificationVisualization.from_config_create(
                config, iterator_name
            )

        # Default classification function is to check for NaNs
        classification_function = lambda x: ~np.isnan(x.astype(float))  # pylint: disable=C3001

        if function_name := method_options.get("classification_model_function_name"):
            external_python_function = method_options["external_python_module_function"]
            classification_function = get_module_attribute(external_python_function, function_name)

        classifier = from_config_create_classifier(
            config, classifier_name, len(model.parameters.to_list())
        )
        return cls(
            model=model,
            result_description=result_description,
            global_settings=global_settings,
            num_sample_points=num_sample_points,
            num_model_calls=num_model_calls,
            classifier=classifier,
            random_sampling_frequency=random_sampling_frequency,
            seed=seed,
            visualization_obj=visualization_obj,
            classification_function=classification_function,
        )

    def core_run(self):
        """Evaluate the samples on model and classify them."""
        # Essential
        print_classification()

        # use active learning
        if self.classifier.is_active:
            _logger.info("Active classification.")
            samples = self.parameters.draw_samples(self.num_sample_points)
            classification_data = np.full(shape=len(samples), fill_value=MISSING_LABEL)
            for i in range(self.num_model_calls // self.classifier.batch_size):
                query_idx = self.classifier.train(samples, classification_data)
                if self.random_sampling_frequency and i % self.random_sampling_frequency == 0:
                    query_idx = self._select_random_samples(classification_data, query_idx)
                classification_data[query_idx.reshape(-1, 1)] = self.binarize(samples[query_idx])

                if self.visualization_obj:
                    self.visualization_obj.plot_decision_boundary(
                        self.classified_outputs,
                        self.samples,
                        self.classifier,
                        self.parameters.names,
                        i,
                    )
                if i % 10 == 0:
                    _logger.info(
                        "Active iteration %i, model calls %i", i, len(self.classified_outputs)
                    )
        else:
            _logger.info("Classification.")
            samples = self.parameters.draw_samples(self.num_model_calls)
            classification_data = self.binarize(samples)
            self.classifier.train(samples, classification_data)

    def _select_random_samples(self, classification_data, query_idx):
        """Select random samples.

        Args:
            classification_data (np.array): Classification of all sample points
            query_idx (np.array): array with the next sample indices

        Returns:
            np.array: new query_idx
        """
        not_predicted_idx = np.argwhere(np.isnan(classification_data))
        idx = np.array(
            list(
                filter(
                    lambda x: x not in query_idx,
                    not_predicted_idx,
                )
            )
        )
        np.random.shuffle(idx)
        return idx[: self.classifier.batch_size].flatten()

    def post_run(self):
        """Analyze the results."""
        if self.result_description:
            results = {"classified_outputs": self.classified_outputs, "input_samples": self.samples}
            if self.result_description["write_results"]:
                write_results(
                    results,
                    self.global_settings["output_dir"],
                    self.global_settings["experiment_name"],
                )

        # plot decision boundary for the trained classifier
        if self.visualization_obj:
            self.visualization_obj.plot_decision_boundary(
                self.classified_outputs,
                self.samples,
                self.classifier,
                self.parameters.names,
                "final",
            )

        # save checkpoint
        self.classifier.save(
            self.global_settings["output_dir"],
            self.global_settings["experiment_name"] + "_classifier",
        )

    def _evaluate_model(self, samples):
        """Evaluate model.

        Args:
            samples (np.array): Samples where to evaluate the model

        Returns:
            np.array: model output
        """
        output = self.model.evaluate(samples)["mean"]
        self.samples = np.row_stack((self.samples, samples))
        return output

    def binarize(self, samples):
        """Classify the output.

        Args:
            samples (np.array): Samples where to evaluate the model

        Returns:
            np.array: classified output
        """
        output = self.classification_function(self._evaluate_model(samples))
        self.classified_outputs = np.row_stack((self.classified_outputs, output)).astype(int)
        return output
