import numpy as np
from pyDOE import lhs
from .iterator import Iterator
from pqueens.models.model import Model
#from pqueens.variables.variables import Variables
from .scale_samples import scale_samples

class LHSIterator(Iterator):
    """ Basic LHS Iterator to enable Latin Hypercube sampling

    Attributes:
        model (model):        Model to be evaluated by iterator
        seed  (int):          Seed for random number generation
        num_samples (int):    Number of samples to compute
        num_iterations (int): Number of optimization iterations of design
        samples (np.array):   Array with all samples
        outputs (np.array):   Array with all model outputs

    """
    def __init__(self, model, seed, num_samples, num_iterations):
        super(LHSIterator, self).__init__(model)
        self.seed = seed
        self.num_samples = num_samples
        self.num_iterations = num_iterations
        self.samples = None
        self.outputs = None

    @classmethod
    def from_config_create_iterator(cls, config, iterator_name=None, model=None):
        """ Create LHS iterator from problem description

        Args:
            config (dict): Dictionary with QUEENS problem description

        Returns:
            iterator: LHSIterator object

        """
        if iterator_name is None:
            method_options = config["method"]["method_options"]
        else:
            method_options = config[iterator_name]["method_options"]
        if model is None:
            model_name = method_options["model"]
            model = Model.from_config_create_model(model_name, config)
        return cls(model, method_options["seed"], method_options["num_samples"],
                   method_options["num_iterations"])

    def eval_model(self):
        """ Evaluate the model """
        return self.model.evaluate()

    def pre_run(self):
        """ Generate samples for subsequent LHS analysis """
        np.random.seed(self.seed)

        distribution_info = self.model.get_parameter_distribution_info()
        numparams = len(distribution_info)
        # create latin hyper cube samples in unit hyper cube
        hypercube_samples = lhs(numparams, self.num_samples,
                                'maximin', iterations=self.num_iterations)
        # scale and transform samples according to the inverse cdf
        self.samples = scale_samples(hypercube_samples, distribution_info)


    def core_run(self):
        """ Run LHS Analysis on model """

        self.model.update_model_from_sample_batch(self.samples)

        self.outputs = self.eval_model()


    def post_run(self):
        """ Analyze the results """

        print("Size of inputs {}".format(self.samples.shape))
        print("Inputs {}".format(self.samples))
        print("Size of outputs {}".format(self.outputs.shape))
        print("Outputs {}".format(self.outputs))
