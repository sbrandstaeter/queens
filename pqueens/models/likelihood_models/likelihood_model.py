"""Module to define likelihood functions."""

import abc

from pqueens.models import from_config_create_model
from pqueens.models.model import Model
from pqueens.utils.experimental_data_reader import ExperimentalDataReader


class LikelihoodModel(Model):
    """Base class for likelihood models.

    Attributes:
        forward_model (obj): Forward model on which the likelihood model is based
        coords_mat (np.array): Row-wise coordinates at which the observations were recorded
        time_vec (np.array): Vector of observation times
        y_obs (np.array): Corresponding experimental data vector to coords_mat
        output_label (str): Name of the experimental outputs (column label in csv-file)
        coord_labels (lst): List with coordinate labels for (column labels in csv-file)
    """

    def __init__(
        self,
        model_name,
        forward_model,
        coords_mat,
        time_vec,
        y_obs,
        output_label,
        coord_labels,
    ):
        """Initialize the likelihood model.

        Args:
            model_name (str): Name of the underlying model in input file
            forward_model (obj): Forward model that is evaluated during the likelihood evaluation
            coords_mat (np.array): Matrix of observation coordinates (new coordinates row-wise)
            time_vec (np.array): Vector containing time stamps for each observation
            y_obs (np.array): Matrix with row-wise observation vectors
            output_label (str): Output label name of the observations
            coord_labels (list): List of coordinate label names. One name per column in coord_mat
        """
        super().__init__(model_name)
        self.forward_model = forward_model
        self.coords_mat = coords_mat
        self.time_vec = time_vec
        self.y_obs = y_obs
        self.output_label = output_label
        self.coord_labels = coord_labels

    @staticmethod
    def get_base_attributes_from_config(model_name, config):
        """Get the attributes of the base class from config.

        Args:
            model_name (str): Name of the model in the input file
            config (dict): Config of the QUEENS run

        Returns:
            forward_model (obj): Forward model that is evaluated during the likelihood evaluation
            coords_mat (np.array): Matrix of observation coordinates (new coordinates row-wise)
            time_vec (np.array): Vector containing time stamps for each observation
            y_obs (np.array): Matrix with row-wise observation vectors
            output_label (str): Output label name of the observations
            coord_labels (list): List of coordinate label names. One name per column in coord_mat
        """
        model_options = config[model_name]
        forward_model_name = model_options.get("forward_model_name")
        forward_model = from_config_create_model(forward_model_name, config)

        experimental_data_reader_name = model_options.get("experimental_data_reader_name")
        experimental_data_reader = (
            ExperimentalDataReader.from_config_create_experimental_data_reader(
                config, experimental_data_reader_name
            )
        )
        (
            y_obs,
            coords_mat,
            time_vec,
            _,
            _,
            coord_labels,
            output_label,
        ) = experimental_data_reader.get_experimental_data()

        return forward_model, coords_mat, time_vec, y_obs, output_label, coord_labels

    @abc.abstractmethod
    def evaluate(self, samples):
        """Evaluate model with current set of input samples.

        Args:
            samples (np.ndarray): Input samples
        """

    @abc.abstractmethod
    def grad(self, samples, upstream_gradient):
        r"""Evaluate gradient of model w.r.t. current set of input samples.

        Consider current model f(x) with input samples x, and upstream function g(f). The provided
        upstream gradient is :math:`\frac{\partial g}{\partial f}` and the method returns
        :math:`\frac{\partial g}{\partial f} \frac{df}{dx}`.

        Args:
            samples (np.array): Input samples
            upstream_gradient (np.array): Upstream gradient function evaluated at input samples
                                          :math:`\frac{\partial g}{\partial f}`

        Returns:
            gradient (np.array): Gradient w.r.t. current set of input samples
                                 :math:`\frac{\partial g}{\partial f} \frac{df}{dx}`
        """
