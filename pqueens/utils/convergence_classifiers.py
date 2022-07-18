"""Classifiers for use in convergence classification"""
from pathlib import Path

from sklearn.svm import SVC, SVR

from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from skactiveml.classifier import SklearnClassifier
from skactiveml.pool import UncertaintySampling

import numpy as np
from typing import Union, Any
import pickle

import warnings

warnings.filterwarnings("ignore")


class Classifier:
    """
    Base class for a classifier.

    Attributes:
        n_params: number of parameters of the solver
        _clf: classifier, e.g. sklearn.svm.SVR
    """
    _clf: Any

    def __init__(self, n_params: int):
        self.n_params = n_params

    def train(self, X_train: np.ndarray, y_true: np.ndarray):
        """
        Train the underlying _clf classifier.

        Args:
            X_train: array with training samples, size: (n_samples, n_params)
            y_true: vector with corresponding training labels, size: (n_samples)

        Returns:
            None or query index for active sampling
        """
        self._clf.fit(X_train, y_true)

    def predict(self, X_test: np.ndarray) -> Union[int, np.ndarray]:
        """
        Perform prediction on given parameter combination, 1 or more samples.
        Args:
            X_test: array of parameter combinations, size: (n_params) for one point, (n_samples, n_params) for multiple points

        Returns:
            y_test: prediction value or vector, size: 1 for one point or (n_samples) for multiple points
        """
        return np.round(self._clf.predict(X_test))

    def save(self, path, file_name):
        pkl_file = Path(path) / (file_name + ".pkl")
        with open(pkl_file, 'wb') as file:
            pickle.dump(self._clf, file)

    def load(self, path, file_name):
        pkl_file = Path(path) / (file_name + ".pkl")
        with open(pkl_file, 'rb') as file:
            self._clf = pickle.load(file)


class SvmClassifier(Classifier):
    """
    A classifier using SVM to predict convergence given a set of input parameters as features.

    Attributes:
        n_params: number of parameters of the solver
        svm: "svr" or "svc"
        kernel: "linear", "poly", "rbf", "sigmoid" or "precomputed"
    """

    def __init__(self, n_params, svm="svc", kernel="rbf"):
        super(SvmClassifier, self).__init__(n_params)
        if svm == "svr":
            self._clf = SVR(
                kernel=kernel)  # optional hyperparameters: C (higher is smoother), gamma (higher means points must be closer)
        else:
            self._clf = SVC(kernel=kernel, max_iter=1e5)


class NNClassifier(Classifier):
    """
    A multi-layer perceptron classifier.

    Attributes:
        n_params: number of parameters of the solver
        solver: solver to use for MLPClassifier, e.g. "adam"
        alpha: alpha parameter for MLPClassifier, default in scikit-learn is 1e-4
        hidden_layer_sizes: list of number of neurons in each layer
    """

    def __init__(self, n_params, solver='adam', alpha=1e-5, hidden_layer_sizes=(32, 32)):
        super(NNClassifier, self).__init__(n_params)
        self._clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation='tanh', solver=solver, alpha=alpha,
                                  random_state=0)


class ActiveLearningClassifier(Classifier):
    """
    Base class for an active learning classifier.

    Attributes:
        _qs: query strategy from skactiveml.pool, e.g. UncertaintySampling
    """
    _qs: UncertaintySampling

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> int:
        """
        Train the underlying _clf classifier.

        Args:
            X_train: array with training samples, size: (n_samples, n_params)
            y_train: vector with corresponding training labels, size: (n_samples)

        Returns:
            query_idx: sample index in X_train to query next
        """
        self._clf.fit(X_train, y_train)
        query_idx = self._qs.query(X=X_train, y=y_train, clf=self._clf, batch_size=1)
        return query_idx

    def predict(self, X_test: np.ndarray) -> Union[int, np.ndarray]:
        return self._clf.predict(X_test)


class GPActiveLearningClassifier(ActiveLearningClassifier):
    """
    A pool-based Gaussian Process active learning classifier.

    Attributes:
        n_params: number of parameters for the solver
    """

    def __init__(self, n_params: int):
        super(GPActiveLearningClassifier, self).__init__(n_params)
        self._qs = UncertaintySampling(method='entropy', random_state=0)
        self._clf = SklearnClassifier(GaussianProcessClassifier(random_state=0), classes=range(n_params))


class DeepActiveClassifier(ActiveLearningClassifier):
    """
    A pool-based neural network active learning classifier.

    Attributes:
        n_params: number of parameters for the solver
        solver: solver to use for MLPClassifier, e.g. "adam"
        alpha: alpha parameter for MLPClassifier, default in scikit-learn is 1e-4
        hidden_layer_sizes: list of number of neurons in each layer
    """

    def __init__(self, n_params, solver='adam', alpha=1e-5, hidden_layer_sizes=(128, 128)):
        super(DeepActiveClassifier, self).__init__(n_params)
        self._qs = UncertaintySampling(method='entropy', random_state=0)
        self._clf = SklearnClassifier(
            MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation='relu', solver=solver, alpha=alpha,
                          random_state=0), classes=range(n_params))
