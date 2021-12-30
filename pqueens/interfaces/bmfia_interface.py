"""Interface for Bayesian multi-fidelity inverse analysis."""

import logging
import multiprocessing as mp
from multiprocessing import Pool

import numpy as np

from pqueens.regression_approximations.regression_approximation import RegressionApproximation

from .interface import Interface

_logger = logging.getLogger(__name__)


class BmfiaInterface(Interface):
    """Interface class for Bayesian multi-fidelity inverse analysis.

    Interface for grouping the outputs of several simulation models with
    identical model inputs to one multi-fidelity data point in the multi-
    fidelity space.

    The BmfiaInterface is basically a version of the
    approximation_interface class that allows for vectorized mapping and
    implicit function relationships by treating every coordinate point (not input point)
    as an individual regression model.

    Attributes:
        config (dict): Dictionary with problem description (input file)
        approx_name (str): Name of the used approximation model
        probabilistic_mapping_obj_lst (lst): List of probabilistic mapping objects which models the
                                             probabilistic dependency between high-fidelity model,
                                             low-fidelity models and informative input features for
                                             each coordinate tuple of
                                             :math: `y_{lf} x y_{hf} x gamma_i` individually.

    Returns:
        BMFMCInterface (obj): Instance of the BMFMCInterface
    """

    def __init__(self, config, approx_name):
        """Instantiate a BMFIA interface."""
        self.config = config
        self.approx_name = approx_name
        self.probabilistic_mapping_obj_lst = []

    def map(self, Z_LF, support='y', full_cov=False):
        r"""Map the lf features to a probabilistic response for the hf model.

        Calls the probabilistic mapping and predicts the mean and variance,
        respectively covariance, for the high-fidelity model, given the inputs
        Z_LF.

        Args:
            Z_LF (np.array): Low-fidelity feature vector that contains the corresponding Monte-Carlo
                             points on which the probabilistic mapping should be evaluated.
                             Dimensions: Rows: different multi-fidelity vector/points
                             (each row is one multi-fidelity point).
                             Columns: different model outputs/informative features.
            full_cov (bool): Boolean that returns full covariance matrix (True) or variance (False)
                             along with the mean prediction
            support (str): Support/variable for which we predict the mean and (co)variance. For
                            `support=f` the Gaussian process predicts w.r.t. the latent function
                            `f`. For the choice of `support=y` we predict w.r.t. to the
                            simulation/experimental output `y`,
                            which introduces the additional variance of the observation noise.

        Returns:
            mean_Y_HF_given_Z_LF (np.array): Vector of mean predictions
                                             :math:`\mathbb{E}_{f^*}[p(y_{HF}^*|f^*,z_{LF}^*,
                                             \mathcal{D}_{f})]` for the HF model given the
                                             low-fidelity feature input. Different HF predictions
                                             per row. Each row corresponds to one multi-fidelity
                                             input vector in
                                             :math:`\Omega_{y_{lf}\times\gamma_i}`.

            var_Y_HF_given_Z_LF (np.array): Vector of variance predictions :math:`\mathbb{V}_{
                                            f^*}[p(y_{HF}^*|f^*,z_{LF}^*,\mathcal{D}_{f})]` for the
                                            HF model given the low-fidelity feature input.
                                            Different HF predictions
                                            per row. Each row corresponds to one multi-fidelity
                                            input vector in
                                            :math:`\Omega_{y_{lf}\times\gamma_i}`.
        """
        if not self.probabilistic_mapping_obj_lst:
            raise RuntimeError(
                "The probabilistic mapping has not been initialized, cannot continue!"
            )

        mean_Y_HF_given_Z_LF = []
        var_Y_HF_given_Z_LF = []

        assert len(self.probabilistic_mapping_obj_lst) == Z_LF.T.shape[0], (
            "The length of the list with probabilistic mapping objects "
            "must agree with the row numbers in Z_LF.T (coordinate dimension)! Abort..."
        )

        # Note: Z_LF might be a 3d tensor here
        # Dims Z_LF: gamma_dim x num_samples x coord_dim
        # Dims Z_LF.T: coord_dim x num_samples x gamma_dim --> iterate over coord_dim
        for z_test_per_coordinate, probabilistic_mapping_obj in zip(
            Z_LF.T, self.probabilistic_mapping_obj_lst
        ):
            if z_test_per_coordinate.ndim > 1:
                output = probabilistic_mapping_obj.predict(
                    z_test_per_coordinate, support=support, full_cov=full_cov
                )
            else:
                output = probabilistic_mapping_obj.predict(
                    np.atleast_2d(z_test_per_coordinate).T, support=support, full_cov=full_cov
                )

            mean_Y_HF_given_Z_LF.append(output["mean"].squeeze())
            var_Y_HF_given_Z_LF.append(output["variance"].squeeze())

        mean = np.atleast_2d(np.array(mean_Y_HF_given_Z_LF)).T
        variance = np.atleast_2d(np.array(var_Y_HF_given_Z_LF)).T
        return mean, variance

    def build_approximation(self, Z_LF_train, Y_HF_train):
        r"""Build the probabilistic regression models.

        Build and train the probabilistic mapping objects based on the
        training inputs :math:`\mathcal{D}_f={Y_{HF},Z_{LF}}` per coordinate
        point / measurement point in the inverse problem.

        Args:
            Z_LF_train (np.array): Training inputs for probabilistic mapping.
                                   Rows: Samples, Columns: Coordinates
            Y_HF_train (np.array): Training outputs for probabilistic mapping.
                                   Rows: Samples, Columns: Coordinates

        Returns:
            None
        """
        assert (
            Z_LF_train.T.shape[0] == Y_HF_train.T.shape[0]
        ), "Dimension of Z_LF_train and Y_HF_train do not agree! Abort ..."

        # Instantiate list of probabilistic mapping
        self.probabilistic_mapping_obj_lst = [
            RegressionApproximation.from_config_create(
                self.config, self.approx_name, np.atleast_2d(z_lf), np.atleast_2d(y_hf).T
            )
            for (z_lf, y_hf) in (zip(Z_LF_train.T, Y_HF_train.T))
        ]

        # prepare parallel pool for training
        num_processors_available = mp.cpu_count()
        num_coords = Z_LF_train.T.shape[0]
        num_processors_for_job = min(num_processors_available, num_coords)

        _logger.info(
            "Run generation and training of probabilistic surrogates in parallel "
            f"on {num_processors_for_job} processors..."
        )

        # Init multi-processing pool
        with Pool(processes=num_processors_for_job) as pool:
            # Actual parallel training of the models
            optimized_hyper_params_lst = pool.map(
                _optimize_hyper_params, self.probabilistic_mapping_obj_lst
            )

        # set the optimized hyper-parameters for probabilistic regression model
        for optimized_state_dict, mapping in zip(
            optimized_hyper_params_lst, self.probabilistic_mapping_obj_lst
        ):
            mapping.set_state(optimized_state_dict)


def _optimize_hyper_params(probabilistic_mapping):
    """Train one probabilistic surrogate model.

    Args:
        probabilsitic_mapping (obj): Instantiated but untrained probabilistic mapping

    Returns:
        optimized_mapping_state_dict (dict): Dictionary with optimized state of the trained
                                             probabilsitic regression model
    """
    probabilistic_mapping.train()
    optimized_mapping_state_dict = probabilistic_mapping.get_state()
    return optimized_mapping_state_dict
