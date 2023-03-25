"""Surrogate model class."""

import logging

import numpy as np

import pqueens.visualization.surrogate_visualization as qvis
from pqueens.interfaces import from_config_create_interface
from pqueens.iterators import from_config_create_iterator
from pqueens.models import from_config_create_model

from .model import Model

_logger = logging.getLogger(__name__)


class DataFitSurrogateModel(Model):
    """Surrogate model class.

    Attributes:
        interface (interface):          Approximation interface.
        subordinate_model: TODO_doc
        subordinate_iterator: TODO_doc
        testing_iterator: TODO_doc
        eval_fit: TODO_doc
        error_measures: TODO_doc
        nash_sutcliffe_efficiency: TODO_doc
    """

    def __init__(
        self,
        model_name,
        interface,
        subordinate_model,
        subordinate_iterator,
        testing_iterator,
        eval_fit,
        error_measures,
        nash_sutcliffe_efficiency,
    ):
        """Initialize data fit surrogate model.

        Args:
            model_name (string):        Name of model
            interface (interface):      Interface to simulator
            subordinate_model (model):  Model the surrogate is based on
            subordinate_iterator (Iterator): Iterator to evaluate the subordinate
                                             model with the purpose of getting
                                             training data
            testing_iterator (Iterator): Iterator to evaluate the subordinate
                                         model with the purpose of getting
                                         testing data
            eval_fit (str):                 How to evaluate goodness of fit
            error_measures (list):          List of error measures to compute
            nash_sutcliffe_efficiency (bool): true if Nash-Sutcliffe efficiency should be evaluated
        """
        super().__init__(model_name)
        self.interface = interface
        self.subordinate_model = subordinate_model
        self.subordinate_iterator = subordinate_iterator
        self.testing_iterator = testing_iterator
        self.eval_fit = eval_fit
        self.error_measures = error_measures
        self.nash_sutcliffe_efficiency = nash_sutcliffe_efficiency

    @classmethod
    def from_config_create_model(cls, model_name, config):
        """Create data fit surrogate model from problem description.

        Args:
            model_name (string): Name of model
            config (dict):       Dictionary containing problem description

        Returns:
            data_fit_surrogate_model: Instance of DataFitSurrogateModel
        """
        # get options
        model_options = config[model_name]
        interface_name = model_options["interface_name"]

        subordinate_model_name = model_options.get("subordinate_model_name", None)
        subordinate_iterator_name = model_options["subordinate_iterator_name"]
        testing_iterator_name = model_options.get("testing_iterator_name", None)

        eval_fit = model_options.get("eval_fit", None)
        error_measures = model_options.get("error_measures", None)

        # create subordinate model
        if subordinate_model_name:
            subordinate_model = from_config_create_model(subordinate_model_name, config)
        else:
            subordinate_model = None

        # create subordinate iterator
        subordinate_iterator = from_config_create_iterator(
            config, subordinate_iterator_name, subordinate_model
        )

        testing_iterator, nash_sutcliffe_efficiency = cls._setup_testing_iterator(
            testing_iterator_name, config, model_options, subordinate_model
        )

        # create interface
        interface = from_config_create_interface(interface_name, config)

        # visualization
        qvis.from_config_create(config, model_name=model_name)

        return cls(
            model_name,
            interface,
            subordinate_model,
            subordinate_iterator,
            testing_iterator,
            eval_fit,
            error_measures,
            nash_sutcliffe_efficiency,
        )

    def evaluate(self, samples):
        """Evaluate model with current set of variables.

        Args:
            samples: TODO_doc

        Returns:
            np.array: Results corresponding to current set of variables
        """
        if not self.interface.is_initialized():
            self.build_approximation()

        self.response = self.interface.evaluate(samples)
        return self.response

    def grad(self, samples, upstream):
        """Evaluate gradient of model with current set of samples.

        Args:
            samples (np.array): Evaluated samples
            upstream (np.array): Upstream gradient
        """
        raise NotImplementedError(
            "Gradient method is not implemented in `data_fit_surrogate_model`."
        )

    def build_approximation(self):
        """Build underlying approximation."""
        self.subordinate_iterator.run()

        # get samples and results
        x_train, y_train = self._get_data_set(self.subordinate_iterator)

        # train regression model on the data
        self.interface.build_approximation(x_train, y_train)

        # plot
        qvis.surrogate_visualization_instance.plot(self.interface)

        if self.eval_fit == "kfold":
            error_measures = self.eval_surrogate_accuracy_cv(
                x_test=x_train, y_test=y_train, k_fold=5, measures=self.error_measures
            )
            for measure, error in error_measures.items():
                _logger.info("Error %s is: %s", measure, error)
        # TODO check that final surrogate is on all points

        if self.testing_iterator:
            self.testing_iterator.run()

            x_test, y_test = self._get_data_set(self.testing_iterator)

            error_measures = self.eval_surrogate_accuracy(x_test, y_test, self.error_measures)
            for measure, error in error_measures.items():
                _logger.info("Error %s is: %s", measure, error)

    def eval_surrogate_accuracy(self, x_test, y_test, measures):
        """Evaluate the accuracy of the surrogate model based on test set.

        Evaluate the accuracy of the surrogate model using the provided
        error metrics.

        Args:
            x_test (np.array):  Test inputs
            y_test (np.array):  Test outputs
            measures (list):    List with desired error metrics

        Returns:
            dict: Dictionary with proving error metrics
        """
        if not self.interface.is_initialized():
            raise RuntimeError("Cannot compute accuracy on uninitialized model")

        response = self.interface.evaluate(x_test)
        y_prediction = response['mean'].reshape((-1, 1))

        error_info = {}
        if measures is not None:
            error_info = self.compute_error_measures(y_test, y_prediction, measures)

        if self.nash_sutcliffe_efficiency is True:
            error_info["nash_sutcliffe_efficiency"] = self.compute_nash_sutcliffe_efficiency(
                y_test, y_prediction
            )
        return error_info

    def eval_surrogate_accuracy_cv(self, x_test, y_test, k_fold, measures):
        """Compute k-fold cross-validation error.

        Args:
            x_test (np.array):       Input array
            y_test (np.array):       Output array
            k_fold (int):       Split dataset in `k_fold` subsets for cv
            measures (list):    List with desired error metrics

        Returns:
            dict:y with error measures and corresponding error values
        """
        if not self.interface.is_initialized():
            raise RuntimeError("Cannot compute accuracy on uninitialized model")

        response_cv = self.interface.cross_validate(x_test, y_test, k_fold)
        y_prediction = np.reshape(np.array(response_cv), (-1, 1))
        error_info = self.compute_error_measures(y_test, y_prediction, measures)

        return error_info

    def compute_error_measures(self, y_test, y_posterior_mean, measures):
        """Compute error measures.

        Compute based on difference between predicted and actual values.

        Args:
            y_test (ndarray): Output values from testing data set
            y_posterior_mean (ndarray): Posterior mean values of the GP
            measures (list):   Dictionary with desired error measures

        Returns:
            dict: Dictionary with error measures and corresponding error values
        """
        error_measures = {}
        for measure in measures:
            error_measures[measure] = self.compute_error(y_test, y_posterior_mean, measure)
        return error_measures

    @staticmethod
    def compute_error(y_test, y_posterior_mean, measure):
        """Compute error for given a specific error measure.

        Args:
            y_test (ndarray): Output values from testing data set
            y_posterior_mean (ndarray): Posterior mean values of the GP
            measure (str):     Desired error metric

        Returns:
            float: Error based on desired metric
        """
        return {
            "sum_squared": np.sum((y_test - y_posterior_mean) ** 2),
            "mean_squared": np.mean((y_test - y_posterior_mean) ** 2),
            "root_mean_squared": np.sqrt(np.mean((y_test - y_posterior_mean) ** 2)),
            "sum_abs": np.sum(np.abs(y_test - y_posterior_mean)),
            "mean_abs": np.mean(np.abs(y_test - y_posterior_mean)),
            "abs_max": np.max(np.abs(y_test - y_posterior_mean)),
        }.get(measure, NotImplementedError("Desired error measure is unknown!"))

    @staticmethod
    def compute_nash_sutcliffe_efficiency(y_test, y_posterior_mean):
        r"""Compute Nash-Sutcliffe model efficiency.

        .. math::
            NSE = 1-\frac{\sum_{i=1}^{N}(e_{i}-s_{i})^2}{\sum_{i=1}^{N}(e_{i}-\bar{e})^2}

        Args:
            y_test (ndarray): Output values from testing data set
            y_posterior_mean (ndarray): Posterior mean values of the GP

        Returns:
            efficiency (float): Nash-Sutcliffe model efficiency
        """
        if len(y_test) == len(y_posterior_mean):
            y_posterior_mean, y_test = np.array(y_posterior_mean), np.array(y_test)
            if y_test.shape != y_posterior_mean.shape:
                y_posterior_mean = y_posterior_mean.transpose()

            mean_observed = np.nanmean(y_test)
            numerator = np.nansum((y_test - y_posterior_mean) ** 2)
            denominator = np.nansum((y_test - mean_observed) ** 2)
            efficiency = 1 - (numerator / denominator)
            return efficiency

        else:
            _logger.warning("Evaluation and simulation lists does not have the same length.")
            return np.nan

    @staticmethod
    def _get_data_set(iterator):
        """Get input and output from iterator.

        Args:
            iterator (pqueens.iterators.Iterator): iterator where to get input and output from

        Returns:
            x (ndarray): input (samples)
            y (ndarray): output (response)
        """
        if hasattr(iterator, 'samples'):
            x = iterator.samples
        else:
            raise AttributeError(
                f'Your iterator {type(iterator).__name__} has no samples and, thus, cannot be used '
                f'for training or testing a surrogate model.'
            )

        if hasattr(iterator, 'output'):
            y = iterator.output['mean']
        else:
            raise AttributeError(
                f'Your iterator {type(iterator).__name__} has no output data and, thus, cannot be '
                f'used for training or testing a surrogate model.'
            )

        return x, y

    @classmethod
    def _setup_testing_iterator(
        cls, testing_iterator_name, config, model_options, subordinate_model
    ):
        """Set up the testing iterator.

        Args:
            config (dict): dictionary containing problem description
            model_options (dict): model options
            subordinate_model (model): model to use
            testing_iterator_name (str): name of the iterator which should be configured

        Returns:
            testing_iterator (iterator): testing iterator
            nash_sutcliffe_efficiency (bool): true if Nash-Sutcliffe efficiency should be evaluated
        """
        if testing_iterator_name:
            testing_iterator = from_config_create_iterator(
                config, testing_iterator_name, subordinate_model
            )
            nash_sutcliffe_efficiency = model_options.get("nash_sutcliffe_efficiency", False)
        else:
            testing_iterator = None
            nash_sutcliffe_efficiency = False
        return testing_iterator, nash_sutcliffe_efficiency
