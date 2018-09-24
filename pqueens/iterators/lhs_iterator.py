import numpy as np
from pyDOE import lhs
from .iterator import Iterator
from pqueens.models.model import Model
from pqueens.utils.process_outputs import process_ouputs
from pqueens.utils.process_outputs import write_results
from .scale_samples import scale_samples

class LHSIterator(Iterator):
    """ Basic LHS Iterator to enable Latin Hypercube sampling

    Attributes:
        model (model):        Model to be evaluated by iterator
        seed  (int):          Seed for random number generation
        num_samples (int):    Number of samples to compute
        num_iterations (int): Number of optimization iterations of design
        result_description (dict):  Description of desired results
        samples (np.array):   Array with all samples
        outputs (np.array):   Array with all model outputs

    """
    def __init__(self, model, seed, num_samples, num_iterations,
                 result_description, global_settings):
        super(LHSIterator, self).__init__(model, global_settings)
        self.seed = seed
        self.num_samples = num_samples
        self.num_iterations = num_iterations
        self.result_description = result_description
        self.samples = None
        self.output = None

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

        result_description = method_options.get("result_description", None)
        global_settings = config.get("global_settings", None)


        return cls(model, method_options["seed"],
                   method_options["num_samples"],
                   method_options.get("num_iterations", 10),
                   result_description,
                   global_settings)

    def eval_model(self):
        """ Evaluate the model """
        return self.model.evaluate()

    def pre_run(self):
        """ Generate samples for subsequent LHS analysis """
        np.random.seed(self.seed)

        parameters = self.model.get_parameter()
        num_inputs = 0
        # get random Variables
        random_variables = parameters.get("random_variables", None)
        # get number of rv
        if random_variables is not None:
            num_rv = len(random_variables)
        num_inputs += num_rv
        # get random fields
        random_fields = parameters.get("random_fields", None)
        if random_fields is not None:
            raise RuntimeError("LHS Sampling is currentyl not implemented in conjunction with random fields.")

        # loop over random variables to generate samples
        distribution_info = []
        for _, rv in random_variables.items():
            # get appropriate random number generator
            temp = {}
            temp["distribution"] = rv["distribution"]
            temp["distribution_parameter"] = rv["distribution_parameter"]
            distribution_info.append(temp)

        # create latin hyper cube samples in unit hyper cube
        hypercube_samples = lhs(num_inputs, self.num_samples,
                                'maximin', iterations=self.num_iterations)
        # scale and transform samples according to the inverse cdf
        self.samples = scale_samples(hypercube_samples, distribution_info)


    def core_run(self):
        """ Run LHS Analysis on model """

        self.model.update_model_from_sample_batch(self.samples)

        self.output = self.eval_model()


    def post_run(self):
        """ Analyze the results """
        if self.result_description is not None:
            results = process_ouputs(self.output, self.result_description)
            if self.result_description["write_results"] is True:
                write_results(results,
                              self.global_settings["output_dir"],
                              self.global_settings["experiment_name"])
        #else:
        print("Size of inputs {}".format(self.samples.shape))
        print("Inputs {}".format(self.samples))
        print("Size of outputs {}".format(self.output['mean'].shape))
        print("Outputs {}".format(self.output['mean']))
