import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol
from pqueens.models.model import Model
from .iterator import Iterator
import random
from pqueens.utils.process_outputs import write_results

# TODO deal with non-uniform input distribution

class SaltelliSALibIterator(Iterator):
    """ Saltelli SALib iterator

        This class essentially provides a wrapper around the SALib librabry

    Attributes:
        seed (int):                         Seed for random number generator
        num_samples (int):                  Number of samples
        calc_second_order (bool):           Calculate second-order sensitivities
        num_bootstrap_samples (int):        Number of bootstrap samples
        confidence_level (float):           The confidence interval level
        samples (np.array):                 Array with all samples
        output (dict)                       Dict with all outputs corresponding to
                                            samples
        salib_problem (dict):               Problem definition for SALib
        num_params (int):                   Number of parameters
        sensitivity_incides (dict):         Dictionary with sensitivity indices
    """
    def __init__(self, model, seed, num_samples, calc_second_order,
                 num_bootstrap_samples, confidence_level, result_description,
                 global_settings):
        """ Initialize Saltelli SALib iterator object

        Args:
            seed (int):                     Seed for random number generation
            num_samples (int):              Number of desired (random) samples
            calc_second_order (bool):       Calculate second-order sensitivities
            num_bootstrap_samples (int):    Number of bootstrap samples
            confidence_level (float):       The confidence interval level
            result_description (dict):      Dictionary with desired result description
        """
        super(SaltelliSALibIterator, self).__init__(model, global_settings)

        self.seed = seed
        self.num_samples = num_samples
        self.calc_second_order = calc_second_order
        self.num_bootstrap_samples = num_bootstrap_samples
        self.confidence_level = confidence_level
        self.result_description = result_description

        self.samples = None
        self.output = None
        self.salib_problem = None
        self.num_params = None
        self.sensitivity_incides = None


    @classmethod
    def from_config_create_iterator(cls, config, model=None):
        """ Create Saltelli SALib iterator from problem description

        Args:
            config (dict): Dictionary with QUEENS problem description

        Returns:
            iterator: Saltelli SALib iterator object

        """
        method_options = config["method"]["method_options"]

        if model is None:
            model_name = method_options["model"]
            model = Model.from_config_create_model(model_name, config)

        return cls(model, method_options["seed"],
                   method_options["num_samples"],
                   method_options["calc_second_order"],
                   method_options["num_bootstrap_samples"],
                   method_options["confidence_level"],
                   method_options.get("result_description", None),
                   config["global_settings"])

    def eval_model(self):
        """ Evaluate the model """
        return self.model.evaluate()

    def pre_run(self):
        """ Generate samples for subsequent analysis and update model """
        np.random.seed(self.seed)
        random.seed(self.seed)
        parameter_info = self.model.get_parameter()

        # setup SALib problem dict
        names = []
        bounds = []
        dists = []
        self.num_params = 0
        for key, value in parameter_info.items():
            names.append(key)
            max_temp = value["distribution_parameter"][1]
            min_temp = value["distribution_parameter"][0]
            bounds.append([min_temp, max_temp])
            dist = self.__get_sa_lib_distribution_name(value["distribution"])
            dists.append(dist)
            self.num_params += 1

        self.salib_problem = {
            'num_vars' : self.num_params,
            'names'    : names,
            'bounds'   : bounds,
            'dists'    : dists
        }
        self.samples = saltelli.sample(self.salib_problem, self.num_samples,
                                       self.calc_second_order)

    def get_all_samples(self):
        """ Return all samples """
        return self.samples

    def core_run(self):
        """ Run Analysis on model """

        #print("Samples :{}".format(self.samples))
        #exit()
        self.model.update_model_from_sample_batch(self.samples)
        self.output = self.eval_model()

        # do actual sensitivity analysis
        self.sensitivity_incides = sobol.analyze(self.salib_problem,
                                                 np.reshape(self.output['mean'], (-1)),
                                                 calc_second_order=self.calc_second_order,
                                                 num_resamples=self.num_bootstrap_samples,
                                                 conf_level=self.confidence_level,
                                                 print_to_console=False)

    def post_run(self):
        """ Analyze the results """
        results = self.process_results()
        if self.result_description is not None:
            if self.result_description["write_results"] is True:
                write_results(results, self.global_settings["output_dir"],
                              self.global_settings["experiment_name"])
            else:
                self.print_results(results)


    def print_results(self, results):
        """ Function to print results """

        S = results["sensitivity_incides"]
        parameter_names = results["parameter_names"]  #self.model.get_parameter_names()
        title = 'Parameter'
        print('%s   S1       S1_conf    ST    ST_conf' % title)
        j = 0
        for name in parameter_names:
            print('%s %f %f %f %f' % (name + '       ', S['S1'][j], S['S1_conf'][j],
                                      S['ST'][j], S['ST_conf'][j]))
            j = j+1

        if results["second_order"]:
            print('\n%s_1 %s_2    S2      S2_conf' % (title, title))
            for j in range(self.num_params):
                for k in range(j + 1, self.num_params):
                    print("%s %s %f %f" % (parameter_names[j] + '            ', parameter_names[k] + '      ',
                                           S['S2'][j, k], S['S2_conf'][j, k]))

    def __get_sa_lib_distribution_name(self, distribution_name):
        """ Convert QUEENS distribution name to SALib distribution name

        Args:
            distribution_name (string): Name of distribution

        Returns:
            string: Name of distribution in SALib
        """
        sa_lib_distribution_name = ''

        if distribution_name == 'uniform':
            sa_lib_distribution_name = 'unif'
        elif distribution_name == 'normal':
            sa_lib_distribution_name = 'norm'
        elif distribution_name == 'lognormal':
            sa_lib_distribution_name = 'lognorm'
        else:
            valid_dists = ['uniform', 'normal', 'lognormal']
            raise ValueError('Distributions: choose one of %s' %
                             ", ".join(valid_dists))
        return sa_lib_distribution_name

    def process_results(self):
        """ Write all results to self contained dictionary """

        results = {}
        results["parameter_names"] = self.model.get_parameter_names()
        results["sensitivity_incides"] = self.sensitivity_incides
        results["second_order"] = self.calc_second_order

        return results
