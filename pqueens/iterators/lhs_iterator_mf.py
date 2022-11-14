"""Multi-fideliy latin hypercube sampling."""

import numpy as np
from pyDOE import lhs

from pqueens.models import from_config_create_model
from pqueens.models.multifidelity_model import MultifidelityModel

from .iterator import Iterator


# TODO add test cases for mf LHS iterator
class MFLHSIterator(Iterator):
    """Multi-Fidelity LHS Iterator to enable Latin Hypercube sampling.

    Multi-Fidelity LHS Iterator with the purpose to generate multi-fidelity
    experimental design for multi-fidelity models. Currently there are two modes,
    independent desings for each level or nested designs where each design is
    a subset of the next higher level.

    Attributes:
        model (model):        multi-fidelity model comprising sub-models
        seed  (int):          Seed for random number generation
        num_samples (list):   List of number of samples to compute on each level
        num_iterations (int): Number of optimization iterations of design
        mode (str):           Mode of sampling (nested/independent)
        samples (list):       List of arrays with all samples
        outputs (list):       List of dicts with all model outputs
    """

    def __init__(self, model, seed, num_samples, num_iterations, mode, global_settings):
        """Initialise multi-fidelity iterator.

        Args:
            model (model):        multi-fidelity model comprising sub-models
            seed  (int):          Seed for random number generation
            num_samples (list):   List of number of samples to compute on each level
            num_iterations (int): Number of optimization iterations of design
            mode (str):           Mode of sampling (nested/independent)
            global_settings (dict, optional): Settings for the QUEENS run.
        """
        super().__init__(model, global_settings)
        if type(self.model) is not MultifidelityModel:
            raise RuntimeError("MFLHS Iterator requires a multi-fidelity model")

        self.seed = seed
        self.num_samples = num_samples
        self.num_iterations = num_iterations
        self.samples = []
        self.outputs = []
        self.mode = mode

    @classmethod
    def from_config_create_iterator(cls, config, iterator_name, model=None):
        """Create LHS iterator from problem description.

        Args:
            config (dict):       Dictionary with QUEENS problem description
            iterator_name (str): Name of iterator to identify right section
                                 in options dict (optional)
            model (model):       Model to use (optional)

        Returns:
            iterator: MFLHSIterator object
        """
        method_options = config[iterator_name]
        print("Method options {}".format(method_options))
        if model is None:
            model_name = method_options["model"]
            model = from_config_create_model(model_name, config)
        return cls(
            model,
            method_options["seed"],
            method_options["num_samples"],
            method_options["num_iterations"],
            method_options["mode"],
            config["global_settings"],
        )

    def pre_run(self):
        """Generate samples for subsequent LHS analysis."""
        np.random.seed(self.seed)
        distribution_info = self.model.get_parameter_distribution_info()
        numparams = len(distribution_info)

        if len(self.num_samples) != self.model.num_levels:
            raise RuntimeError("Number of levels does not match lenght of samples list")

        if self.mode == "nested":
            for i in range(len(self.num_samples)):
                if i == 0:
                    # create latin hyper cube samples in unit hyper cube
                    hypercube_samples = lhs(
                        numparams, self.num_samples[0], 'maximin', iterations=self.num_iterations
                    )
                    # scale and transform samples according to the inverse cdf
                    self.samples.append(self.parameters_inverse_transform(hypercube_samples))
                else:
                    self.samples.append(
                        self.select_random_subset(self.samples[i - 1], self.num_samples[i])
                    )
        elif self.mode == "independent":
            for i in range(len(self.num_samples)):
                hypercube_samples = lhs(
                    numparams, self.num_samples[i], 'maximin', iterations=self.num_iterations
                )
                # scale and transform samples according to the inverse cdf
                self.samples.append(self.parameters_inverse_transform(hypercube_samples))
        else:
            raise ValueError("Mode must be either 'nested' or 'independent' ")

    def core_run(self):
        """Run LHS Analysis on model."""
        self.model.set_response_mode("bypass_lofi")

        for i in range(0, self.model.num_levels):
            self.model.set_hifi_model_index(i)
            self.outputs.append(self.model.evaluate(self.samples[i]))

    def post_run(self):
        """Analyze the results."""
        for i in range(self.model.num_levels):
            print("Size of inputs in LHS{}".format(self.samples[i].shape))
            print("Inputs {}".format(self.samples[i]))
            print("Size of outputs {}".format(self.outputs[i]['mean'].shape))
            print("Outputs {}".format(self.outputs[i]['mean']))

    def select_random_subset(self, samples, subset_size):
        """Select a subset of provided samples and return it.

        Args:
            samples (np.array): Array with samples
            subset_size  (int):  Size of subset to generate

        Returns:
            np.array: Subset of samples
        """
        num_samples = samples.shape[0]
        subset_indices = np.random.choice(num_samples, subset_size, replace=False)
        subset = samples[subset_indices, :]
        return subset
