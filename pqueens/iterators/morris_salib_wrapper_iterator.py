import random
import numpy as np
from SALib.analyze import morris as morris_analyzer
from SALib.sample import morris
from pqueens.models.model import Model
from .iterator import Iterator


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

        samples (np.array):         Samples at which model is evaluated

        outputs (np.array):         Results at samples

        salib_problem (dict):       Dictionary with SALib problem description

        sensitivity_indices (dict): Dictionary with all sensitivity indices

"""

    def __init__(self, model, num_trajectories, local_optimization,
                 num_optimal_trajectories, grid_jump, num_levels, seed,
                 confidence_level, num_bootstrap_samples):
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

        """
        super(MorrisSALibIterator, self).__init__(model)
        self.num_trajectories = num_trajectories
        self.local_optimization = local_optimization
        self.num_optimal_trajectories = num_optimal_trajectories
        self.num_levels = num_levels
        self.seed = seed
        self.confidence_level = confidence_level
        self.num_bootstrap_samples = num_bootstrap_samples
        self.grid_jump = grid_jump

        self.num_params = None
        self.samples = None
        self.outputs = None
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
                   method_options["num_bootstrap_samples"])

    def eval_model(self):
        """ Evaluate the model """
        return self.model.evaluate()

    def pre_run(self):
        """ Generate samples for subsequent analysis and update model """
        random.seed(self.seed)
        np.random.seed(self.seed)

        parameter_info = self.model.get_parameter()

        # setup SALib problem dict
        names = []
        bounds = []
        self.num_params = 0
        for key, value in parameter_info.items():
            names.append(key)
            max_temp = value["max"]
            min_temp = value["min"]
            bounds.append([min_temp, max_temp])
            if "distribution" in value:
                raise ValueError("Parameters must not have probability distributions")
            self.num_params += 1

        self.salib_problem = {
            'num_vars' : self.num_params,
            'names'    : names,
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

        self.outputs = self.eval_model()

        self.si = morris_analyzer.analyze(self.salib_problem,
                                          self.samples,
                                          self.outputs,
                                          num_resamples=self.num_bootstrap_samples,
                                          conf_level=self.confidence_level,
                                          print_to_console=False,
                                          num_levels=self.num_levels,
                                          grid_jump=self.grid_jump)

    def post_run(self):
        """ Analyze the results """
        self.__print_results()

    def __print_results(self):
        """ Print results to screen """

        print("{0:<30} {1:>10} {2:>10} {3:>15} {4:>10}".format(
            "Parameter",
            "Mu_Star",
            "Mu",
            "Mu_Star_Conf",
            "Sigma"))

        for j in range(self.num_params):
            print("{0!s:30} {1!s:10} {2!s:10} {3!s:15} {4!s:10}".format(
                self.si['names'][j],
                self.si['mu_star'][j],
                self.si['mu'][j],
                self.si['mu_star_conf'][j],
                self.si['sigma'][j]))
