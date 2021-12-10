from pqueens.regression_approximations.regression_approximation import RegressionApproximation

from .interface import Interface


class BmfmcInterface(Interface):
    """Interface for grouping the outputs of several simulation with identical
    input to one data point. The BmfmcInterface is basically a version of the
    approximation_interface class that allows for vectorized mapping and
    implicit function relationships.

    Attributes:
        variables (dict): dictionary with variables (not used at the moment!)
        config (dict): Dictionary with problem description (input file)
        approx_name (str): Name of the used approximation model
        probabilistic_mapping_obj (obj): Instance of the probabilistic mapping which models the
                                         probabilistic dependency between high-fidelity model,
                                         low-fidelity models and informative input features.

    Returns:
        BMFMCInterface (obj): Instance of the BMFMCInterface
    """

    def __init__(self, config, approx_name, variables=None):
        # TODO we should think about using the parent class interface here
        self.variables = variables  # TODO: This is not used at the moment!
        self.config = config
        self.approx_name = approx_name
        self.probabilistic_mapping_obj = None

    def map(self, Z_LF, support='y', full_cov=False):
        """Calls the probabilistic mapping and predicts the mean and variance
        for the high-fidelity model, given the inputs Z_LF.

        Args:
            Z_LF (np.array): low-fidelity feature vector that contains the corresponding Monte-Carlo
                              points on which the probabilistic mapping should be evaluated

        Returns:
            mean_Y_HF_given_Z_LF (np.array): Vector of mean predictions
                                             :math:`\\mathbb{E}_{f^*}[p(y_{HF}^*|f^*,z_{LF}^*,
                                             \\mathcal{D}_{f})]` for the HF model given the
                                             low-fidelity feature input
            var_Y_HF_given_Z_LF (np.array): Vector of variance predictions :math:`\\mathbb{V}_{
                                            f^*}[p(y_{HF}^*|f^*,z_{LF}^*,\\mathcal{D}_{f})]` for the
                                            HF model given the low-fidelity feature input
        """

        if self.probabilistic_mapping_obj is None:
            raise RuntimeError(
                "The probabilistic mapping has not been properly initialized, cannot continue!"
            )

        output = self.probabilistic_mapping_obj.predict(Z_LF, support=support, full_cov=full_cov)
        mean_Y_HF_given_Z_LF = output["mean"]
        var_Y_HF_given_Z_LF = output["variance"]
        return mean_Y_HF_given_Z_LF, var_Y_HF_given_Z_LF

    def build_approximation(self, Z_LF_train, Y_HF_train):
        """Build and train the probabilistic mapping based on the training
        inputs :math:`\\mathcal{ D}_f={Y_{HF},Z_{LF}}`

        Args:
            Z_LF_train (np.array): Training inputs for probabilistic mapping
            Y_HF_train (np.array): Training outputs for probabilistic mapping

        Returns:
            None
        """
        self.probabilistic_mapping_obj = RegressionApproximation.from_config_create(
            self.config, self.approx_name, Z_LF_train, Y_HF_train
        )

        self.probabilistic_mapping_obj.train()
