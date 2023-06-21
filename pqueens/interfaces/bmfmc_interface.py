"""Interface for grouping outputs with inputs."""
from pqueens.interfaces.interface import Interface
from pqueens.models import from_config_create_model


class BmfmcInterface(Interface):
    """Interface for grouping outputs with inputs.

    Interface for grouping the outputs of several simulations with identical
    input to one data point. The *BmfmcInterface* is basically a version of the
    *approximation_interface* class, that allows vectorized mapping and
    implicit function relationships.

    Attributes:
        config (dict): Dictionary with problem description (input file).
        approx_name (str): Name of the used approximation model.
        probabilistic_mapping_obj (obj): Instance of the probabilistic mapping,
                                         which models the probabilistic
                                         dependency between high-fidelity
                                         model, low-fidelity models and
                                         informative input features.

    Returns:
        BMFMCInterface (obj): Instance of the BMFMCInterface
    """

    def __init__(self, config, approx_name):
        """Initialize the interface.

        Args:
            config (dict): Dictionary with problem description
            approx_name (str): Name of the approximation model
        """
        self.config = config
        self.approx_name = approx_name
        self.probabilistic_mapping_obj = None

    def evaluate(self, samples, support='y', full_cov=False, gradient_bool=False):
        r"""Predict on probabilistic mapping.

        Call the probabilistic mapping and predict the mean and variance
        for the high-fidelity model, given the inputs *z_lf* (called samples here).

        Args:
            samples (np.array): Low-fidelity feature vector *z_lf* that contains the corresponding
                                Monte-Carlo points on which the probabilistic mapping should
                                be evaluated
            support (str): Support/variable for which we predict the mean and (co)variance. For
                           *support=f*  the Gaussian process predicts w.r.t. the latent function
                           *f*. For the choice of *support=y* we predict w.r.t. the
                           simulation/experimental output *y*
            gradient_bool (bool): Flag to determine whether the gradient of the function at
                                  the evaluation point is expected (*True*) or not (*False*)

        Returns:
            mean_Y_HF_given_Z_LF (np.array): Vector of mean predictions
              :math:`\mathbb{E}_{f^*}[p(y_{HF}^*|f^*,z_{LF}^*, \mathcal{D}_{f})]`
              for the HF model given the low-fidelity feature input
            var_Y_HF_given_Z_LF (np.array): Vector of variance predictions
              :math:`\mathbb{V}_{f^*}[p(y_{HF}^*|f^*,z_{LF}^*,\mathcal{D}_{f})]`
              for the HF model given the low-fidelity feature input
        """
        if self.probabilistic_mapping_obj is None:
            raise RuntimeError(
                "The probabilistic mapping has not been properly initialized, cannot continue!"
            )
        if gradient_bool:
            raise NotImplementedError(
                "The gradient response is not implemented for this interface. Please set "
                "`gradient_bool=False`. Abort..."
            )

        output = self.probabilistic_mapping_obj.predict(samples, support=support, full_cov=full_cov)
        mean_Y_HF_given_Z_LF = output["mean"]
        var_Y_HF_given_Z_LF = output["variance"]
        return mean_Y_HF_given_Z_LF, var_Y_HF_given_Z_LF

    def build_approximation(self, Z_LF_train, Y_HF_train):
        r"""Build and train the probabilistic mapping.

        Based on the training inputs
        :math:`\mathcal{D}_f={Y_{HF},Z_{LF}}`.

        Args:
            Z_LF_train (np.array): Training inputs for probabilistic mapping
            Y_HF_train (np.array): Training outputs for probabilistic mapping
        """
        surrogate_model_name = self.config[self.approx_name]['surrogate_model_name']
        self.probabilistic_mapping_obj = from_config_create_model(surrogate_model_name, self.config)
        self.probabilistic_mapping_obj.setup(Z_LF_train, Y_HF_train)
        self.probabilistic_mapping_obj.train()
