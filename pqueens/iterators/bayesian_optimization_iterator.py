import numpy as np
from pqueens.iterators.iterator import Iterator
from pqueens.models.model import Model
from pqueens.utils.process_outputs import write_results

import gpflow
from gpflowopt.domain import ContinuousParameter
from gpflowopt.bo import BayesianOptimizer
from gpflowopt.optim import StagedOptimizer, MCOptimizer, SciPyOptimizer
#from gpflowopt.optim import SciPyBasinHoppingOptimizer, SciPyDifferentialEvoOptimizer
from gpflowopt.design import LatinHyperCube
from gpflowopt.acquisition import ExpectedImprovement
#from gpflowopt.acquisition import LowerConfidenceBound


class BayesOptIterator(Iterator):
    """ Iterator for Bayesian Optimization

    Attributes:
        model (model):              Model to be evaluated by iterator
        seed  (int):                Seed for random number generation
        num_initial_samples (int):  Number of samples in initial experimental design
        num_params (int):           Number of parameters
        domain ():                  Domain
        num_iter (int):             Number of iterations to run Bayesian optimizer
        use_ard (bool):             Flag whether or not to use ARD in covariance
                                    function
        results (dict):             Container for results from gpflowopt
        result_description (dict):  Dict with description for result processing

    """
    def __init__(self, model, seed, num_iter, use_ard, num_initial_samples,
                 result_description, global_settings):
        super(BayesOptIterator, self).__init__(model, global_settings)
        self.seed = seed
        self.num_iter = num_iter
        self.use_ard = use_ard
        self.num_initial_samples = num_initial_samples

        self.inputs = None
        self.results = None
        self.results_bo = None
        self.result_description = result_description

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
        self.num_params = num_inputs


        gpflow_params = []
        # TODO what happens for non-uniform input??
        for param_name, param_info in random_variables.items():
            gpflow_params.append(ContinuousParameter(param_name,
                                                     param_info["distribution_parameter"][0],
                                                     param_info["distribution_parameter"][1]))
        # for some reason we need to use np.sum here instead on just sum
        self.domain = np.sum(gpflow_params)

    @classmethod
    def from_config_create_iterator(cls, config, iterator_name=None, model=None):
        """ Create Bayesian Optimization iterator from problem description

        Args:
            config (dict): Dictionary with QUEENS problem description

        Returns:
            iterator: BayesOptIterator object

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

        return cls(model, method_options["seed"], method_options["num_iter"],
                   method_options["use_ard"], method_options["num_initial_samples"],
                   result_description, global_settings)

    def eval_model(self):
        """ Evaluate the model """
        return self.model.evaluate()

    def prep_and_eval_model(self, X):
        """ Interface function to create a compatible interface to GPflowOpt

        Args:
            X (np.array): Array with to be evaluated inputs

        Returns:
            np.array: Array with results corresponding to inputs
        """
        if self.inputs is None:
            self.inputs = X
        else:
            self.inputs = np.append(self.inputs, X, axis=0)

        self.model.update_model_from_sample_batch(X)
        results = self.eval_model()
        if self.results is None:
            self.results = results["mean"]
        else:
            self.results = np.append(self.results, results["mean"], axis=0)

        return results["mean"]

    def core_run(self):
        """  Run Bayesian Optimization """
        # Run initial experimental design
        np.random.seed(self.seed)
        lhd = LatinHyperCube(self.num_initial_samples, self.domain)
        X = lhd.generate()
        Y = self.prep_and_eval_model(X)

        # setup Gaussian process
        model = gpflow.gpr.GPR(X, Y, gpflow.kernels.Matern52(self.num_params,
                                                             ARD=self.use_ard))

        model.kern.lengthscales.transform = gpflow.transforms.Log1pe(1e-3)

        # create the Bayesian optimizer
        # TODO make acquisition function an option
        alpha = ExpectedImprovement(model)
        # alpha = LowerConfidenceBound(model)
        # TODO make optimizer an option
        #acquisition_opt = SciPyBasinHoppingOptimizer(self.domain)
        #acquisition_opt = SciPyDifferentialEvoOptimizer(self.domain)

        acquisition_opt = StagedOptimizer([MCOptimizer(self.domain, 1000),
                                           SciPyOptimizer(self.domain)])


        optimizer = BayesianOptimizer(self.domain, alpha,
                                      optimizer=acquisition_opt,
                                      scaling=False)

        # Run the Bayesian optimization
        with optimizer.silent():
            self.results_bo = optimizer.optimize(self.prep_and_eval_model,
                                                 n_iter=self.num_iter)

    def post_run(self):
        """ Analyze the results """
        if self.result_description is not None:
            if self.result_description["write_results"] is True:
                write_results(self.results_bo,
                              self.global_settings["output_dir"],
                              self.global_settings["experiment_name"])
        else:
            print("Evaluated inputs: {}".format(self.inputs))
            print("Evaluated results: {}".format(self.results))
            print("Summary BO {}".format(self.results_bo))
