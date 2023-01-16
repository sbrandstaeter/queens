"""Class for mapping input variables to responses using a MF approximation."""
import numpy as np

from pqueens.regression_approximations_mf import from_comfig_create_regression_approximators_mf

from .interface import Interface

# TODO add tests


class ApproximationInterfaceMF(Interface):
    """Class for mapping input variables to responses using a MF approximation.

        The ApproximationInterface uses a so-called regression approximation,
        which is just another name for a regression model that is used in this context,
        to avoid confusion and not having to call everything a model.

        For now this interface holds only one approximation object. In the future,
        this could be extended to multiple objects.

    Attributes:
        approximation_config (dict): Config options for approximation.
        approximation (regression_approximation_mf): Approximation object.
        approx_init (bool): Flag whether approximation has been initialized.
    """

    def __init__(self, interface_name, approximation_config):
        """Create interface.

        Args:
            interface_name (string):     Name of interface
            approximation_config (dict): Config options for approximation
        """
        super().__init__(interface_name)
        self.approximation_config = approximation_config
        self.approximation = None
        self.approx_init = False

    @classmethod
    def from_config_create_interface(cls, interface_name, config):
        """Create interface from config dictionary.

        Args:
            interface_name (str):   Name of interface
            config (dict):          Dictionary containing problem description

        Returns:
            interface: Instance of ApproximationInterface
        """
        interface_options = config[interface_name]
        approximation_name = interface_options["approximation"]
        approximation_config = config[approximation_name]
        parameters = config['parameters']

        # initialize object
        return cls(interface_name, approximation_config, parameters)

    # TODO think about introducing general mf-interface ?
    def evaluate(self, samples, gradient_bool=False):
        """Mapping function which calls the regression approximation.

        Args:
            samples (list):         List of variables objects
            gradient_bool (bool): Flag to determine whether the gradient of the function at
                                  the evaluation point is expected (*True*) or not (*False*)

        Returns:
            dict: Dictionary with mean, variance, and possibly posterior samples (*post_samples*)
            at samples
        """
        if not self.approx_init:
            raise RuntimeError("Approximation has not been properly initialized, cannot continue!")

        if gradient_bool:
            raise NotImplementedError(
                "The gradient response is not implemented for this interface. Please set "
                "`gradient_bool=False`. Abort..."
            )

        inputs = []
        for variables in samples:
            params = variables.get_active_variables()
            inputs.append(list(params.values()))

        # get inputs as array and reshape
        num_active_vars = samples[0].get_number_of_active_variables()
        inputs = np.reshape(np.array(inputs), (-1, num_active_vars), order='F')
        output = self.approximation.predict(inputs)
        return output

    def build_approximation(self, Xtrain, Ytrain):
        """Build and train underlying regression model.

        Args:
            Xtrain (list):      List of arrays of Training inputs
            Ytrain (list):      List of arrays of Training outputs
        """
        self.approximation = from_comfig_create_regression_approximators_mf(
            self.approximation_config, Xtrain, Ytrain
        )
        self.approximation.train()
        self.approx_init = True

    def is_initialized(self):
        """Check if the approximation interface is initialized."""
        return self.approx_init
