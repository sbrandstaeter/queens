import random
import numpy as np
from SALib.analyze import morris as morris_analyzer
from SALib.sample import morris
from pqueens.models.model import Model
from .iterator import Iterator


class MorrisSALibIterator(Iterator):
    """ Morris SAlib Wrapper Iterator to compute elementary effects

    References :
    [1] A. Saltelli, M. Ratto, T. Andres, F. Campolongo, T. Cariboni,
        D. Galtelli, M. Saisana, S. Tarantola. "GLOBAL SENSITIVITY ANALYSIS.
        The Primer", 109 - 121, ISBN : 978-0-470-05997-5

    [2] Morris, M. (1991).  "Factorial Sampling Plans for Preliminary
        Computational Experiments."  Technometrics, 33(2):161-174,
        doi:10.1080/00401706.1991.10484804.

    [3] Campolongo, F., J. Cariboni, and A. Saltelli (2007).  "An effective
        screening design for sensitivity analysis of large models."
        Environmental Modelling & Software, 22(10):1509-1518,
        doi:10.1016/j.envsoft.2006.10.004.

    Attributes:

        num_traj (int):             Number of trajectories in the input space

        optim (bool):               True if we want to perform brute-force
                                    optimization from Campolongo. False if we
                                    choose directly the num_traj, in this case
                                    num_traj and num_traj_chosen have to be equal

        num_traj_chosen (int):      Number of trajectories chosen in the design,
                                    with the brute-force optimization from Campolongo.

        num_levels (int):           The number of grid levels

        seed (int):                 Seed for random number generation

        confidence_level (float):   Size of confidence interval

        n_bootstrap_samples (int):  Number of bootstrap samples used to
                                    compute confidence intervals for
                                    sensitivity measures

        num_params (int):           Number of model parameters

        samples (np.array):         Samples at which model is evaluated

        outputs (np.array):         Results at sampling inputs

        sensitivity_indices (dict): Dictionary with all sensitivity indices

"""

    def __init__(self, model, num_traj, optim, num_traj_chosen, grid_jump,
                 num_levels, seed, confidence_level, n_bootstrap_samples):
        """ Initialize MorrisSALibIterator

        Args:
            model (model):             QUEENS model to evaluate

            num_traj (int):            Number of trajectories in the input space

            optim (bool):              True if we want to perform brute-force
                                       optimization from Campolongo. False if we
                                       choose directly the num_traj, in this
                                       case num_traj and num_traj_chosen are
                                       equal

            num_traj_chosen (int):     Number of trajectories chosen in the
                                       design, with the brute-force optimization
                                       from Campolongo

            grid_jump (int):           The grid jump size

            num_levels (int):          The number of grid levels

            seed (int):                Seed for random number generation

            confidence_level (float):  Size of confidence interval to compute
                                       for the sensitivity measures

            num_bootstrap_samples (int): Number of bootstrap samples used to
                                       compute confidence intervals for
                                       sensitivity measures

        """
        super(MorrisSALibIterator, self).__init__(model)
        self.num_traj = num_traj
        self.optim = optim
        self.num_traj_chosen = num_traj_chosen
        self.num_levels = num_levels
        self.seed = seed
        self.confidence_level = confidence_level
        self.num_bootstrap_samples = n_bootstrap_samples
        self.num_params = None
        self.samples = None
        self.outputs = None
        self.sensitivity_indices = {}
        self.grid_jump = grid_jump
        self.salib_problem = {}

    @classmethod
    def from_config_create_iterator(cls, config):
        """ Create MorrisCampolongo iterator from problem description

        Args:
            config (dict): Dictionary with QUEENS problem description

        Returns:
            iterator: MorrisCampolongo iterator object

        """
        method_options = config["method"]["method_options"]
        model_name = method_options["model"]

        model = Model.from_config_create_model(model_name, config)

        return cls(model, method_options["num_traj"],
                   method_options["optim"],
                   method_options["num_traj_chosen"],
                   method_options["grid_jump"],
                   method_options["number_of_levels"],
                   method_options["seed"],
                   method_options["confidence_level"],
                   method_options["number_of_bootstrap_samples"])

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
                                     self.num_traj,
                                     num_levels=self.num_levels,
                                     grid_jump=self.grid_jump,
                                     optimal_trajectories=self.num_traj_chosen,
                                     local_optimization=True)

        print("Sample shape {}".format(self.samples.shape))
        print("Samples {}".format(self.samples))


    def core_run(self):
        """ Run Analysis on model """

        self.model.update_model_from_sample_batch(self.samples)
        self.outputs = self.eval_model()

    def post_run(self):
        """ Analyze the results """
        self.sensitivity_indices = morris_analyzer.analyze(self.salib_problem,
                                                           self.samples,
                                                           self.outputs,
                                                           num_resamples=self.num_bootstrap_samples,
                                                           conf_level=self.confidence_level,
                                                           print_to_console=False,
                                                           num_levels=self.num_levels,
                                                           grid_jump=self.grid_jump)
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
                self.sensitivity_indices['names'][j],
                self.sensitivity_indices['mu_star'][j],
                self.sensitivity_indices['mu'][j],
                self.sensitivity_indices['mu_star_conf'][j],
                self.sensitivity_indices['sigma'][j]))
