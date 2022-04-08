"""Gaussian static likelihood."""
import numpy as np

from pqueens.utils.iterative_averaging_utils import from_config_create_iterative_averaging

from .likelihood_model import LikelihoodModel


class GaussianStaticLikelihood(LikelihoodModel):
    """Gaussian likelihood model with static noise (one hyperparameter).

    Attributes:
        noise_var (float): Current value of the likelihood noise parameter
        nugget_noise_var (float): Lower bound for the likelihood noise parameter
        likelihood_noise_type (str): String encoding the type of likelihood noise model:
                                     Fixed or MAP estimate with Jeffreys prior
        fixed_likelihood_noise_value (float): Value for likelihood noise in case the fixed option
                                              was chosen
        noise_var_iterative_averaging (obj): Iterative averaging object

    Returns:
        Instance of GaussianStaticLikelihood Class
    """

    def __init__(
        self,
        model_name,
        model_parameters,
        nugget_noise_var,
        forward_model,
        coords_mat,
        time_vec,
        y_obs_vec,
        likelihood_noise_type,
        fixed_likelihood_noise_value,
        output_label,
        coord_labels,
        noise_var_iterative_averaging,
    ):
        """Initialize likelihood model.

        Args:
            model_name (str): Model name
            model_parameters (dict): Dictionary with description of uncertain
                                     parameters
            forward_model (obj): Forward model on which the likelihood model is based
            coords_mat (np.array): Row-wise coordinates at which the observations were recorded
            time_vec (np.array): Vector of observation times
            y_obs_vec (np.array): Corresponding experimental data vector to coords_mat
            output_label (str): Name of the experimental outputs (column label in csv-file)
            coord_labels (lst): List with coordinate labels for (column labels in csv-file)
            nugget_noise_var (float): Lower bound for the likelihood noise parameter
            likelihood_noise_type (str): String encoding the type of likelihood noise model:
                                         Fixed or MAP estimate with Jeffreys prior
            fixed_likelihood_noise_value (float): Value for likelihood noise in case the fixed
                                                  option was chosen
            noise_var_iterative_averaging (obj): Iterative averaging object
        """
        super().__init__(
            model_name,
            model_parameters,
            forward_model,
            coords_mat,
            time_vec,
            y_obs_vec,
            output_label,
            coord_labels,
        )
        self.noise_var = None
        self.nugget_noise_var = nugget_noise_var
        self.likelihood_noise_type = likelihood_noise_type
        self.fixed_likelihood_noise_value = fixed_likelihood_noise_value
        self.noise_var_iterative_averaging = noise_var_iterative_averaging

    @classmethod
    def from_config_create_likelihood(
        cls,
        model_name,
        config,
        model_parameters,
        forward_model,
        coords_mat,
        time_vec,
        y_obs_vec,
        output_label,
        coord_labels,
    ):
        """Create Gaussian static likelihood model from problem description.

        Args:
            model_name (str): Name of the likelihood model
            config (dict): Dictionary containing problem description
            model_parameters (dict): Dictionary containing description of model parameters
            forward_model (obj): Forward model on which the likelihood model is based
            coords_mat (np.array): Row-wise coordinates at which the observations were recorded
            time_vec (np.array): Vector of observation times
            y_obs_vec (np.array): Corresponding experimental data vector to coords_mat
            output_label (str): Name of the experimental outputs (column label in csv-file)
            coord_labels (lst): List with coordinate labels for (column labels in csv-file)

        Returns:
            instance of GaussianStaticLikelihood class
        """
        # get options
        model_options = config[model_name]

        # get specifics of gaussian static likelihood model
        likelihood_noise_type = model_options["likelihood_noise_type"]
        fixed_likelihood_noise_value = model_options.get("fixed_likelihood_noise_value")
        nugget_noise_var = model_options.get("nugget_noise_var", 1e-6)

        noise_var_iterative_averaging = model_options.get("noise_var_iterative_averaging", None)
        if noise_var_iterative_averaging:
            noise_var_iterative_averaging = from_config_create_iterative_averaging(
                noise_var_iterative_averaging
            )

        return cls(
            model_name=model_name,
            model_parameters=model_parameters,
            nugget_noise_var=nugget_noise_var,
            forward_model=forward_model,
            coords_mat=coords_mat,
            time_vec=time_vec,
            y_obs_vec=y_obs_vec,
            likelihood_noise_type=likelihood_noise_type,
            fixed_likelihood_noise_value=fixed_likelihood_noise_value,
            output_label=output_label,
            coord_labels=coord_labels,
            noise_var_iterative_averaging=noise_var_iterative_averaging,
        )

    def evaluate(self):
        """Evaluate likelihood with current set of samples.

        Returns:
            log_likelihood (np.array): Vector of log-likelihood values per model input.
        """
        Y_mat = self._update_and_evaluate_forward_model()

        n = self.y_obs_vec.size
        self._update_noise_var(n, Y_mat)

        log_likelihood = []
        for y_vec in Y_mat:

            dist = np.atleast_2d(self.y_obs_vec - y_vec)
            dist_squared = (dist**2).reshape(1, -1)  # squared distances as row vector
            log_likelihood.append(
                -n / 2 * np.log(2 * np.pi)
                - n / 2 * np.log(self.noise_var)
                - 1 / (2 * self.noise_var) * np.sum(dist_squared, axis=1)
            )
            # potentially extent likelihood by Jeffreys prior
            if self.likelihood_noise_type == "jeffreys_prior":
                log_likelihood[-1] = (
                    log_likelihood[-1] + np.log(np.sqrt(2.0)) - 0.5 * np.log(self.noise_var)
                )

        log_likelihood = np.array(log_likelihood)
        return log_likelihood

    def _noise_var_iterative_averaging(self):
        """Average likelihood noise iteratively."""
        self.noise_var = self.noise_var_iterative_averaging.update_average(self.noise_var)

    def _update_noise_var(self, num_obs, Y_mat):
        """Potentially update the static noise variance.

        The noise is computed based on a MAP estimate using a Jeffreys prior on the var. If
        activated, the noise is averaged according to a iterative averaging scheme.

        Args:
            num_obs (int): Number of experimental observations (length of y_obs)
            Y_mat (np.array): Matrix of row-wise simulation output at observation coordinates for
                              input batch X_batch
        """
        # either keep noise level fixed or regulate it with a Jeffreys prior
        if self.likelihood_noise_type == "fixed":
            self.noise_var = self.fixed_likelihood_noise_value
        elif self.likelihood_noise_type == 'jeffreys_prior':
            # the line below is the map estimate for the noise variance with jeffreys prior
            # we calculate the MAP over the entire batch data set
            dist_squared = np.zeros((1, num_obs))
            for y_vec in Y_mat:
                dist = np.atleast_2d(self.y_obs_vec - y_vec)
                dist_squared += (dist**2).reshape(1, -1)  # squared distances as row vector

            self.noise_var = np.sum(dist_squared, axis=1) / (1 + (num_obs * Y_mat.shape[0]))

            # If iterative averaging is desired
            if self.noise_var_iterative_averaging:
                self.noise_var = self.noise_var_iterative_averaging.update_average(self.noise_var)
            print(self.noise_var)
            # Limit noise to minimum value set by the nugget_var
            if self.noise_var < self.nugget_noise_var:
                print(
                    "Calculated likelihood noise variance fell below the nugget-noise variance. "
                    "Resetting likelihood noise variance to nugget_noise..."
                )
                self.noise_var = self.nugget_noise_var

        else:
            raise ValueError(
                f"Please specify a valid likelihood noise type! Your choice of "
                f"{self.likelihood_noise_type} is invalid! Valid options are fixed or"
                f"jeffreys_prior"
            )

    def _update_and_evaluate_forward_model(self):
        """Evaluate forward model.

        Pass the variables update to subordinate simulation model and then
        evaluate the simulation model.

        Returns:
           Y_mat (np.array): Simulation output (row-wise) that corresponds to input batch X_batch
        """
        # Note that the wrapper of the model update needs to called externally such that
        # self.variables is updated
        self.forward_model.variables = self.variables
        n_samples_batch = len(self.variables)  # TODO check if this is generally true
        Y_mat = self.forward_model.evaluate()['mean'][-n_samples_batch:]  # TODO check this

        return Y_mat
