"""Classifiers for use in convergence classification."""
import pickle
from pathlib import Path

import numpy as np
from skactiveml.classifier import SklearnClassifier
from skactiveml.pool import UncertaintySampling
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from pqueens.utils.valid_options_utils import get_option


class Classifier:
    """Classifier wrapper.

    Attributes:
        n_params (int): number of parameters of the solver
        classifier_obj (obj): classifier, e.g. sklearn.svm.SVR
    """

    is_active = False

    def __init__(self, n_params, classifier_obj):
        """Initialise the classifier.

        Args:
            n_params (int): number of parameters
            classifier_obj (obj): classifier, e.g. sklearn.svm.SVR
        """
        self.n_params = n_params
        self.classifier_obj = classifier_obj

    def train(self, x_train, y_train):
        """Train the underlying _clf classifier.

        Args:
            x_train: array with training samples, size: (n_samples, n_params)
            y_train: vector with corresponding training labels, size: (n_samples)
        """
        self.classifier_obj.fit(x_train, y_train)

    def predict(self, x_test):
        """Perform prediction on given parameter combinations.

        Args:
            x_test (np.array): array of parameter combinations (n_samples, n_params)

        Returns:
            y_test: prediction value or vector (n_samples)
        """
        # Binary classification
        return np.round(self.classifier_obj.predict(x_test))

    def save(self, path, file_name):
        """Pickle the classifier.

        Args:
            path (str): Path to export the classifier
            file_name (str): File name without suffix
        """
        pickle_file = Path(path) / (file_name + ".pickle")
        with pickle_file.open('wb') as file:
            pickle.dump(self.classifier_obj, file)

    def load(self, path, file_name):
        """Load pickled the classifier.

        Args:
            path (str): Path to export the classifier
            file_name (str): File name without suffix
        """
        pickle_file = Path(path) / (file_name + ".pickle")
        with pickle_file.open('rb') as file:
            self.classifier_obj = pickle.load(file)


class ActiveLearningClassifier(Classifier):
    """Active learning classifier wrapper.

    Attributes:
        n_params (int): number of parameters of the solver
        classifier_obj (obj): classifier, e.g. sklearn.svm.SVR
        active_sampler_obj: query strategy from skactiveml.pool, e.g. UncertaintySampling
    """

    is_active = True

    def __init__(self, n_params, classifier_obj, batch_size, active_sampler_obj=None):
        """Initialise active learning classifier.

        Args:
            n_params (int): number of parameters of the solver
            classifier_obj (obj): classifier, e.g. sklearn.svm.SVR
            active_sampler_obj (obj): query strategy from skactiveml.pool, e.g. UncertaintySampling
            batch_size (int): Batch size to query the the next samples.
        """
        super().__init__(n_params, SklearnClassifier(classifier_obj, classes=range(2)))
        if active_sampler_obj:
            self.active_sampler_obj = active_sampler_obj
        else:
            self.active_sampler_obj = UncertaintySampling(method='entropy', random_state=0)
        self.batch_size = batch_size

    def train(self, x_train, y_train):
        """Train the underlying _clf classifier.

        Args:
            x_train (np.array): array with training samples, size: (n_samples, n_params)
            y_train (np.array): vector with corresponding training labels, size: (n_samples)

        Returns:
            query_idx (np.array): sample indices in x_train to query next
        """
        self.classifier_obj.fit(x_train, y_train)
        query_idx = self.active_sampler_obj.query(
            X=x_train, y=y_train, clf=self.classifier_obj, batch_size=self.batch_size
        )
        return query_idx


def from_config_create_classifier(config, classifier_name, n_params):
    """Create classifier from config.

    Args:
        config (dict): queens run configuration
        classifier_name (str): name of classifier
        n_params (int): number of parameters

    Returns:
        obj: classifier object
    """
    classifier_options = config[classifier_name]
    classifier_type = classifier_options.pop("type")

    batch_size = None
    if "batch_size" in classifier_options:
        batch_size = classifier_options.pop("batch_size")

    available_options = {
        "svc": SVC,
        "nn": MLPClassifier,
        "gp": GaussianProcessClassifier,
    }

    classifier_obj = get_option(available_options, classifier_type)(**classifier_options)

    if batch_size:
        return ActiveLearningClassifier(n_params, classifier_obj, batch_size)

    return Classifier(n_params, classifier_obj)
