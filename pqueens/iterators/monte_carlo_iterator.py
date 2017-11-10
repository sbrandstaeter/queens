import numpy as np
from .iterator import Iterator
from pqueens.models.model import Model
from pqueens.variables.variables import Variables

class MonteCarloIterator(Iterator):
    """ Basic Monte Carlo Iterator to enable MC sampling

    Attributes:
        model (model):      Model to be evaluated by iterator
        seed  (int):        Seed for random number generation
        num_samples (int):  Number of samples to compute

    """
    def __init__(self, model, seed, num_samples):
        super(MonteCarloIterator, self).__init__(model)
        self.seed = seed
        self.num_samples = num_samples
        self.mc = None

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
        self.mc = np.zeros((self.num_samples, numparams))

        i = 0
        for distribution in distribution_info:
            # get appropriate random number generator
            random_number_generator = getattr(np.random, distribution['distribution'])
            # make a copy
            my_args = list(distribution['distribution_parameter'])
            my_args.extend([self.num_samples])
            self.mc[:, i] = random_number_generator(*my_args)
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
        inputs = self.mc
        self.model.update_model_from_sample_batch(inputs)
        outputs = self.eval_model()

        # Print
        #print("Size of inputs {}".format(inputs.shape))
        print("Inputs {}".format(np.array(inputs)))
        #print("Size of outputs {}".format(outputs.shape))
        print("Outputs {}".format(outputs))


    def __sample_generator(self):
        """ Generator to iterate over samples """
        i = 0
        while i < self.num_samples:
            yield self.mc[i, :]
            i += 1

    def __get_all_samples(self):
        """ Return all samples """
        return self.mc
