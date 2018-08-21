import random
import numpy as np
from SALib.analyze import morris as morris_analyzer
from SALib.sample import morris
from pqueens.models.model import Model
from .iterator import Iterator
from pqueens.utils.process_outputs import write_results



class MorrisSALibIterator(Iterator):
    """ Morris SAlib Wrapper Iterator to compute elementary effects

    Attributes:

        num_trajectories (int):     The number of trajectories to generate

        local_optimization (bool):  Flag whether to use local optimization
                                    according to Ruano et al. (2012)
                                    Speeds up the process tremendously for larger
                                    number of trajectories and num_levels.
                                    If set to ``False`` brute force method is used.

        num_optimal_trajectories (int): The number of optimal trajectories to
                                        sample (between 2 and N)

        num_levels (int):             The number of grid levels

        seed (int):                   Seed for random number generation

        confidence_level (float):     Size of confidence interval

        num_bootstrap_samples (int):  Number of bootstrap samples used to
                                      compute confidence intervals for
                                      sensitivity measures

        num_params (int):           Number of model parameters

        parameter_names (list):     List with parameter names

        samples (np.array):         Samples at which model is evaluated

        output (np.array):          Results at samples

        salib_problem (dict):       Dictionary with SALib problem description

        sensitivity_indices (dict): Dictionary with all sensitivity indices

"""

    def __init__(self, model, num_trajectories, local_optimization,
                 num_optimal_trajectories, grid_jump, num_levels, seed,
                 confidence_level, num_bootstrap_samples, result_description,
                 global_settings):
        """ Initialize MorrisSALibIterator

        Args:
            model (model):             QUEENS model to evaluate

            num_trajectories (int):     The number of trajectories to generate

            local_optimization (bool):  Flag whether to use local optimization
                                        according to Ruano et al. (2012)
                                        Speeds up the process tremendously for larger
                                        number of trajectories and num_levels.
                                        If set to ``False`` brute force method is used.

            num_optimal_trajectories (int): The number of optimal trajectories to
                                            sample (between 2 and N)

            num_levels (int):             The number of grid levels

            seed (int):                   Seed for random number generation

            confidence_level (float):     Size of confidence interval

            num_bootstrap_samples (int):  Number of bootstrap samples used to
                                          compute confidence intervals for
                                          sensitivity measures
            result_description (dict):      Dictionary with desired result description


        """
        super(MorrisSALibIterator, self).__init__(model, global_settings)
        self.num_trajectories = num_trajectories
        self.local_optimization = local_optimization
        self.num_optimal_trajectories = num_optimal_trajectories
        self.num_levels = num_levels
        self.seed = seed
        self.confidence_level = confidence_level
        self.num_bootstrap_samples = num_bootstrap_samples
        self.grid_jump = grid_jump
        self.result_description = result_description

        self.num_params = None
        self.parameter_names = []
        self.samples = None
        self.output = None
        self.salib_problem = {}
        self.si = {}



    @classmethod
    def from_config_create_iterator(cls, config, model=None):
        """ Create MorrisSALibIterator iterator from problem description

        Args:
            config (dict): Dictionary with QUEENS problem description

        Returns:
            iterator: MorrisSALibIterator iterator object

        """
        method_options = config["method"]["method_options"]

        if model is None:
            model_name = method_options["model"]
            model = Model.from_config_create_model(model_name, config)

        if not "num_traj_chosen" in method_options:
            method_options["num_traj_chosen"] = None

        return cls(model, method_options["num_trajectories"],
                   method_options["local_optimization"],
                   method_options["num_optimal_trajectories"],
                   method_options["grid_jump"],
                   method_options["number_of_levels"],
                   method_options["seed"],
                   method_options["confidence_level"],
                   method_options["num_bootstrap_samples"],
                   method_options.get("result_description", None),
                   config["global_settings"])

    def eval_model(self):
        """ Evaluate the model """
        return self.model.evaluate()

    def pre_run(self):
        """ Generate samples for subsequent analysis and update model """
        random.seed(self.seed)
        np.random.seed(self.seed)
        np.random.RandomState(seed=self.seed)

        parameter_info = self.model.get_parameter()

        # setup SALib problem dict
        bounds = []
        self.num_params = 0
        for key, value in parameter_info["random_variables"].items():
            self.parameter_names.append(key)
            max_temp = value["max"]
            min_temp = value["min"]
            bounds.append([min_temp, max_temp])
            if "distribution" in value:
                raise ValueError("Parameters must not have probability distributions")
            self.num_params += 1

        if parameter_info.get("random_fields", None) is not None:
            raise RuntimeError("LHS Sampling is currentyl not implemented in conjunction with random fields.")


        self.salib_problem = {
            'num_vars' : self.num_params,
            'names'    : self.parameter_names,
            'bounds'   : bounds,
            'groups'   : None
            }

        self.samples = morris.sample(self.salib_problem,
                                     self.num_trajectories,
                                     num_levels=self.num_levels,
                                     grid_jump=self.grid_jump,
                                     optimal_trajectories=self.num_optimal_trajectories,
                                     local_optimization=True)



    def core_run(self):
        """ Run Analysis on model """

        self.model.update_model_from_sample_batch(self.samples)

        self.output = self.eval_model()

        self.si = morris_analyzer.analyze(self.salib_problem,
                                          self.samples,
                                          np.reshape(self.output['mean'], (-1)),
                                          num_resamples=self.num_bootstrap_samples,
                                          conf_level=self.confidence_level,
                                          print_to_console=False,
                                          num_levels=self.num_levels,
                                          grid_jump=self.grid_jump)


    def post_run(self):
        """ Analyze the results """
        results = self.process_results()
        if self.result_description is not None:
            if self.result_description["write_results"] is True:
                write_results(results, self.global_settings["output_dir"],
                              self.global_settings["experiment_name"])
            else:
                self.print_results(results)


    def process_results(self):
        """ Write all results to self contained dictionary """

        results = {}
        results["parameter_names"] = self.parameter_names
        results["sensitivity_incides"] = self.si
        return results


    def print_results(self, results):
        """ Print results to screen """

        print("{0:<30} {1:>10} {2:>10} {3:>15} {4:>10}".format(
            "Parameter",
            "Mu_Star",
            "Mu",
            "Mu_Star_Conf",
            "Sigma"))

        for j in range(self.num_params):
            print("{0!s:30} {1!s:10} {2!s:10} {3!s:15} {4!s:10}".format(
                results['sensitivity_incides']['names'][j],
                results['sensitivity_incides']['mu_star'][j],
                results['sensitivity_incides']['mu'][j],
                results['sensitivity_incides']['mu_star_conf'][j],
                results['sensitivity_incides']['sigma'][j]))
