"""Module to define likelihood functions."""

import abc

from pqueens.models.model import Model


class LikelihoodModel(Model):
    """Base class for likelihood models.

    Attributes:
        forward_model (obj): Forward model on which the likelihood model is based
        coords_mat (np.array): Row-wise coordinates at which the observations were recorded
        time_vec (np.array): Vector of observation times
        y_obs_vec (np.array): Corresponding experimental data vector to coords_mat
        output_label (str): Name of the experimental outputs (column label in csv-file)
        coord_labels (lst): List with coordinate labels for (column labels in csv-file)

    Returns:
        Instance of LikelihoodModel class
    """

    def __init__(
        self,
        model_name,
        model_parameters,
        forward_model,
        coords_mat,
        time_vec,
        y_obs_vec,
        output_label,
        coord_labels,
    ):
        """Initialize the likelihood model.

        Args:
            model_name (str): Name of the underlying model in input file
            model_parameters (OrderedDict): Parameters of the underlying model
            forward_model (obj): Forward model that is evaluated during the likelihood evaluation
            coords_mat (np.array): Matrix of observation coordinates (new coordinates row-wise)
            time_vec (np.array): Vector containing time stamps for each observation
            y_obs_vec (np.array): Matrix with row-wise observation vectors
            output_label (str): Output label name of the observations
            coord_labels (list): List of coordinate label names. One name per column in coord_mat
        """
        super().__init__(model_name, model_parameters)
        self.forward_model = forward_model
        self.coords_mat = coords_mat
        self.y_obs_vec = y_obs_vec
        self.output_label = output_label
        self.time_vec = time_vec
        self.coord_labels = coord_labels

    @abc.abstractmethod
    def evaluate(self):
        """Evaluate model with current set of variables."""
