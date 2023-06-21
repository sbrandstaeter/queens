"""Helper classes for Gaussian process prediction for Sobol indices."""
import copy
import logging
import multiprocessing as mp
import time

import numpy as np
import xarray as xr

from pqueens.iterators.sobol_index_gp_uncertainty.utils_prediction import (
    predict_mean,
    sample_realizations,
)

_logger = logging.getLogger(__name__)


class Predictor:
    """Predictor class.

    Predict the output of the Gaussian process used as a surrogate model at all Monte-Carlo samples
    stored in the sample matrices (A, B, AB, BA,...). Predicting either mean sampling realizations
    of the Gaussian process or using the posterior mean of the Gaussian process (if the number of
    GP realizations is set to 1).

    Attributes:
        number_gp_realizations (int): number of Gaussian process realizations
        gp_model (Model): Gaussian process model
        seed_posterior_samples (int): seed for posterior samples
    """

    def __init__(self, gp_model, number_gp_realizations, seed_posterior_samples):
        """Initialize.

        Args:
            gp_model (Model): Gaussian process model
            number_gp_realizations (int): number of Gaussian process realizations
            seed_posterior_samples (int): seed for posterior samples
        """
        self.gp_model = gp_model
        self.number_gp_realizations = number_gp_realizations
        self.seed_posterior_samples = seed_posterior_samples

    @classmethod
    def from_config_create(cls, method_options, gp_model):
        """Create estimator from problem description.

        Args:
            method_options (dict): dictionary with method options
            gp_model (Model): Gaussian process model

        Returns:
            estimator: SobolIndexEstimator
        """
        number_gp_realizations = method_options['number_gp_realizations']
        seed_posterior_samples = method_options.get("seed_posterior_samples", None)
        if number_gp_realizations == 1:
            _logger.info('Number of realizations = 1. Prediction is based on GP mean.')
        else:
            _logger.info('Number of realizations = {}'.format(number_gp_realizations))
        return cls(
            gp_model=gp_model,
            number_gp_realizations=number_gp_realizations,
            seed_posterior_samples=seed_posterior_samples,
        )

    def predict(self, samples, num_procs):
        """Predict output at Monte-Carlo samples.

        Sample realizations of Gaussian process or use the posterior mean of the Gaussian process.

        Args:
            samples (xr.Array): Monte-Carlo samples
            num_procs (int): number of processors

        Returns:
            prediction (xr.Array): predictions
        """
        start_prediction = time.time()

        prediction = self._init_prediction(samples)
        prediction_function, input_list = self._setup_parallelization(samples, prediction)

        # start multiprocessing pool
        pool = mp.get_context("spawn").Pool(num_procs)
        raw_prediction = pool.starmap(prediction_function, input_list)
        pool.close()

        prediction.data = np.array(raw_prediction)

        _logger.info(f'Time for prediction: {time.time() - start_prediction}')
        _logger.debug(f'Prediction: {prediction.values}')
        return prediction

    def _init_prediction(self, samples):
        """Initialize prediction data-array.

        Args:
            samples (xr.DataArray): Monte-Carlo samples

        Returns:
            prediction (xr.DataArray): predictions
        """
        dimensions = ("monte_carlo", "sample_matrix", "gp_realization")
        coordinates = {
            "monte_carlo": samples.coords["monte_carlo"].values,
            "sample_matrix": samples.coords["sample_matrix"].values,
            "gp_realization": np.arange(self.number_gp_realizations),
        }

        prediction = xr.DataArray(
            data=np.empty(
                (
                    samples.coords["monte_carlo"].size,
                    samples.coords["sample_matrix"].size,
                    self.number_gp_realizations,
                )
            ),
            dims=dimensions,
            coords=coordinates,
        )

        return prediction

    def _setup_parallelization(self, samples, prediction):
        """Set up parallelization.

        Args:
            samples (xr.DataArray): Monte-Carlo samples
            prediction (xr.DataArray): predictions

        Returns:
            prediction_function (obj): function object for prediction
            input_list (list): list of input for prediction_function
        """
        # This is a workaround, as the queens python interface can not be pickled.
        gp_model = copy.deepcopy(self.gp_model)
        gp_model.training_iterator = None
        gp_model.testing_iterator = None

        if self.number_gp_realizations == 1:
            prediction_function = predict_mean
            input_list = [
                (samples.loc[dict(monte_carlo=m)].values, gp_model)
                for m in prediction.coords["monte_carlo"]
            ]
        else:
            prediction_function = sample_realizations
            input_list = [
                (
                    samples.loc[dict(monte_carlo=m)].values,
                    gp_model,
                    self.number_gp_realizations,
                    self.seed_posterior_samples,
                )
                for m in prediction.coords["monte_carlo"]
            ]

        return prediction_function, input_list
