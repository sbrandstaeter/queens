from pqueens.regression_approximations.regression_approximation import RegressionApproximation

from .interface import Interface
import numpy as np

import logging

_logger = logging.getLogger(__name__)


class BmfiaInterface(Interface):
    """
    Interface for grouping the outputs of several simulation models with identical model inputs to
    one multi-fidelity data point in the multi-fidelity space:
    .. math::
        \\Omega: y_{hf} x y_{lf} x \\gamma_{i} 

    The BmfiaInterface is basically a version of the
    approximation_interface class that allows for vectorized mapping and
    implicit function relationships by treating every coordinate point (not input point)
    as an individual regression model.

    Attributes:
        variables (dict): dictionary with variables (not used at the moment!)
        config (dict): Dictionary with problem description (input file)
        approx_name (str): Name of the used approximation model
        probabilistic_mapping_obj_lst (lst): List of probabilistic mapping objects which models the
                                             probabilistic dependency between high-fidelity model,
                                             low-fidelity models and informative input features for
                                             each coordinate touple of 
                                             :math: `y_{lf} x y_{hf} x gamma_i` individually.

    Returns:
        BMFMCInterface (obj): Instance of the BMFMCInterface

    """

    def __init__(self, config, approx_name, variables=None):
        # TODO we should think about using the parent class interface here
        self.variables = variables  # TODO: This is not used at the moment!
        self.config = config
        self.approx_name = approx_name
        self.probabilistic_mapping_obj_lst = []

    def map(self, Z_LF, support='y', full_cov=False):
        """
        Calls the probabilistic mapping and predicts the mean and variance,
        respectively covariance, for the high-fidelity model,
        given the inputs Z_LF.

        Args:
            Z_LF (np.array): Low-fidelity feature vector that contains the corresponding Monte-Carlo
                             points on which the probabilistic mapping should be evaluated.
                             Dimensions: Rows: differnt multi-fidelity vector/points
                             (each row is one multi-fidelity point).
                             Columns: different model outputs/informative features.
            full_cov (bool): Boolean that returns full covariance matrix (True) or variance (False)
                             along with the mean prediction
            support (str): Support/variable for which we predict the mean and (co)variance. For
                            `suppoprt=f` the Gaussian process predicts w.r.t. the latent function
                            `f`. For the choice of `support=y` we predict w.r.t. to the
                            simulation/experimental output `y`,
                            which introduces the additional variance of the observation noise.

        Returns:
            mean_Y_HF_given_Z_LF (np.array): Vector of mean predictions
                                             :math:`\\mathbb{E}_{f^*}[p(y_{HF}^*|f^*,z_{LF}^*,
                                             \\mathcal{D}_{f})]` for the HF model given the
                                             low-fidelity feature input. Different HF predictions
                                             per row. Each row corresponds to one multi-fidelity
                                             input vector in
                                             :math:`\\Omega_{y_{lf}\\times\\gamma_i}`.
                                             
            var_Y_HF_given_Z_LF (np.array): Vector of variance predictions :math:`\\mathbb{V}_{
                                            f^*}[p(y_{HF}^*|f^*,z_{LF}^*,\\mathcal{D}_{f})]` for the
                                            HF model given the low-fidelity feature input.
                                            Different HF predictions
                                            per row. Each row corresponds to one multi-fidelity
                                            input vector in
                                            :math:`\\Omega_{y_{lf}\\times\\gamma_i}`.


        """
        if not self.probabilistic_mapping_obj_lst:
            raise RuntimeError(
                "The probabilistic mapping has not been properly initialized, cannot continue!"
            )

        mean_Y_HF_given_Z_LF = []
        var_Y_HF_given_Z_LF = []

        for z_test, probabilistic_mapping_obj in zip(Z_LF.T, self.probabilistic_mapping_obj_lst):
            if z_test.ndim > 1:
                output = probabilistic_mapping_obj.predict(
                    np.atleast_2d(z_test), support=support, full_cov=full_cov
                )
            else:
                output = probabilistic_mapping_obj.predict(
                    np.atleast_2d(z_test).T, support=support, full_cov=full_cov
                )

            mean_Y_HF_given_Z_LF.append(output["mean"].squeeze())
            var_Y_HF_given_Z_LF.append(output["variance"].squeeze())

        mean = np.atleast_2d(np.array(mean_Y_HF_given_Z_LF)).T
        variance = np.atleast_2d(np.array(var_Y_HF_given_Z_LF)).T
        return mean, variance

    def build_approximation(self, Z_LF_train, Y_HF_train):
        """
        Build and train the probabilistic mapping objects based on the
        training inputs :math:`\\mathcal{D}_f={Y_{HF},Z_{LF}}` per
        coordinate point / measurement point in the inverse problem.

        Args:
            Z_LF_train (np.array): Training inputs for probabilistic mapping.
                                   Rows: Samples, Columns: Coordinates
            Y_HF_train (np.array): Training outputs for probabilistic mapping.
                                   Rows: Samples, Columns: Coordinates

        Returns:
            None

        """
        # TODO: make this parallel!
        for num, (z_lf, y_hf) in enumerate(zip(Z_LF_train.T, Y_HF_train.T)):
            _logger.info(f'Training model {num + 1} of {Z_LF_train.T.shape[0]}.')
            self.probabilistic_mapping_obj_lst.append(
                RegressionApproximation.from_config_create(
                    self.config, self.approx_name, z_lf.reshape(-1, 1), y_hf.reshape(-1, 1)
                )
            )
            self.probabilistic_mapping_obj_lst[-1].train()
