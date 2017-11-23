import numpy as np
from .iterator import Iterator
from pqueens.models.model import Model
from pqueens.variables.variables import Variables

class MonteCarloIterator(Iterator):
    """ Basic Monte Carlo Iterator to enable MC sampling

    Attributes:
        model (model):        Model to be evaluated by iterator
        seed  (int):          Seed for random number generation
        num_samples (int):    Number of samples to compute
        samples (np.array):   Array with all samples
        outputs (np.array):   Array with all model outputs
    """
    def __init__(self, model, seed, num_samples):
        super(MonteCarloIterator, self).__init__(model)
        self.seed = seed
        self.num_samples = num_samples
        self.samples = None
        self.outputs = None

    @classmethod
    def from_config_create_iterator(cls, config):
        """ Create MC iterator from problem description

        Args:
            config (dict): Dictionary with QUEENS problem description

        Returns:
            iterator: MonteCarloIterator object

        """
        method_options = config["method"]["method_options"]
        model_name = method_options["model"]

        model = Model.from_config_create_model(model_name, config)
        return cls(model, method_options["seed"], method_options["num_samples"])

    def eval_model(self):
        """ Evaluate the model """
        return self.model.evaluate()

    def pre_run(self):
        """ Generate samples for subsequent MC analysis and update model """
        np.random.seed(self.seed)

        distribution_info = self.model.get_parameter_distribution_info()
        numparams = len(distribution_info)
        self.samples = np.zeros((self.num_samples, numparams))

        i = 0
        for distribution in distribution_info:
            # get appropriate random number generator
            random_number_generator = getattr(np.random, distribution['distribution'])
            # make a copy
            my_args = list(distribution['distribution_parameter'])
            my_args.extend([self.num_samples])
            self.samples [:, i] = random_number_generator(*my_args)
            i += 1


    def core_run(self):
        """  Run Monte Carlo Analysis on model """
        # variant 1
        # inputs = []
        # outputs = []
        # for sample in self.__sample_generator():
        #     my_input = sample
        #     self.model.update_model_from_sample(my_input)
        #     my_output = self.eval_model()
        #     inputs.append(my_input)
        #     outputs.append(my_output)

        # variant 2
        self.model.update_model_from_sample_batch(self.samples)

        self.outputs = self.eval_model()

    def post_run(self):
        """ Analyze the results """

        print("Size of inputs {}".format(self.samples.shape))
        print("Inputs {}".format(self.samples))
        print("Size of outputs {}".format(self.outputs.shape))
        print("Outputs {}".format(self.outputs))
