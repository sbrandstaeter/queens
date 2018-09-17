import numpy as np
from .iterator import Iterator
from pqueens.models.model import Model
from pqueens.utils.process_outputs import process_ouputs
from pqueens.utils.process_outputs import write_results
from pqueens.randomfields.univariate_field_generator_factory import UniVarRandomFieldGeneratorFactory
from pqueens.utils.input_to_random_variable import get_distribution_object
from pqueens.utils.input_to_random_variable import get_random_samples


class MonteCarloIterator(Iterator):
    """ Basic Monte Carlo Iterator to enable MC sampling

    Attributes:
        model (model):              Model to be evaluated by iterator
        seed  (int):                Seed for random number generation
        num_samples (int):          Number of samples to compute
        result_description (dict):  Description of desired results
        samples (np.array):         Array with all samples
        outputs (np.array):         Array with all model outputs
    """
    def __init__(self, model, seed, num_samples, result_description, global_settings):
        super(MonteCarloIterator, self).__init__(model, global_settings)
        self.seed = seed
        self.num_samples = num_samples
        self.result_description = result_description
        self.samples = None
        self.output = None

    @classmethod
    def from_config_create_iterator(cls, config, iterator_name=None, model=None):
        """ Create MC iterator from problem description

        Args:
            config (dict): Dictionary with QUEENS problem description

        Returns:
            iterator: MonteCarloIterator object

        """
        print(config.get("experiment_name"))
        if iterator_name is None:
            method_options = config["method"]["method_options"]
        else:
            method_options = config[iterator_name]["method_options"]
        if model is None:
            model_name = method_options["model"]
            model = Model.from_config_create_model(model_name, config)

        result_description = method_options.get("result_description", None)
        global_settings = config.get("global_settings", None)

        return cls(model,
                   method_options["seed"],
                   method_options["num_samples"],
                   result_description,
                   global_settings)

    def eval_model(self):
        """ Evaluate the model """
        return self.model.evaluate()

    def pre_run(self):
        """ Generate samples for subsequent MC analysis and update model """
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

        num_eval_locations = []

        if random_fields is not None:
            for _, rf in random_fields.items():
                dim = rf["dimension"]
                eval_locations_list = rf.get("eval_locations", None)
                eval_locations = np.array(eval_locations_list).reshape(-1, dim)
                temp = eval_locations.shape[0]
                num_eval_locations.append(temp)
                num_inputs += temp


        self.samples = np.zeros((self.num_samples, num_inputs))
        # loop over random variables to generate samples
        i = 0
        for _, rv in random_variables.items():
            self.samples[:, i] = get_random_samples(rv, self.num_samples)
            i += 1

        # loop over random fields to generate samples
        field_num = 0
        if random_fields is not None:
            for _, rf in random_fields.items():
                print("rf corrstruct {}".format(rf.get("corrstruct")))
                # create appropriate random field generator
                my_field_generator = UniVarRandomFieldGeneratorFactory.create_new_random_field_generator(
                    get_distribution_object(rf),
                    rf.get("dimension"),
                    rf.get("corrstruct"),
                    rf.get("corr_length"),
                    rf.get("energy_frac"),
                    np.array(rf.get("field_bbox")),
                    rf.get("num_terms_per_dim"),
                    rf.get("total_terms"))

                dim = rf["dimension"]
                eval_locations_list = rf.get("eval_locations", None)
                eval_locations = np.array(eval_locations_list).reshape(-1, dim)
                my_stoch_dim = my_field_generator.get_stoch_dim()

                my_vals = np.zeros((self.num_samples, eval_locations.shape[0]))
                for i in range(self.num_samples):
                    xi = np.random.randn(my_stoch_dim, 1)
                    my_vals[i, :] = my_field_generator.evaluate_field_at_location(eval_locations, xi)

                self.samples[:, num_rv+field_num:num_rv+field_num+len(eval_locations)] = my_vals
                field_num += 1


    def core_run(self):
        """  Run Monte Carlo Analysis on model """

        self.model.update_model_from_sample_batch(self.samples)

        self.output = self.eval_model()

    def post_run(self):
        """ Analyze the results """
        if self.result_description is not None:
            results = process_ouputs(self.output, self.result_description, self.samples)
            if self.result_description["write_results"] is True:
                write_results(results,
                              self.global_settings["output_dir"],
                              self.global_settings["experiment_name"])
        else:
            print("Size of inputs {}".format(self.samples.shape))
            print("Inputs {}".format(self.samples))
            print("Size of outputs {}".format(self.output['mean'].shape))
            print("Outputs {}".format(self.output['mean']))
