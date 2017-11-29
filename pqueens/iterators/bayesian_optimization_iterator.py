import numpy as np
from pqueens.iterators.iterator import Iterator
from pqueens.models.model import Model

import gpflow
from gpflowopt.domain import ContinuousParameter
from gpflowopt.bo import BayesianOptimizer
from gpflowopt.design import LatinHyperCube
from gpflowopt.acquisition import ExpectedImprovement

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
        results ():                 Container for results from gpflowopt

    """
    def __init__(self, model, seed, num_iter, use_ard, num_initial_samples):
        super(BayesOptIterator, self).__init__(model)
        self.seed = seed
        self.num_iter = num_iter
        self.use_ard = use_ard
        self.num_initial_samples = num_initial_samples

        self.results = None

        #self.domain = None
        parameter_info = self.model.get_parameter()
        self.num_params = len(parameter_info)

        gpflow_params = []
        for param_name, param_info in parameter_info.items():
            gpflow_params.append(ContinuousParameter(param_name,
                                                     param_info["distribution_parameter"][0],
                                                     param_info["distribution_parameter"][1]))
        # for some reason we need to use np.sum here instead on just sum
        self.domain = np.sum(gpflow_params)

    @classmethod
    def from_config_create_iterator(cls, config):
        """ Create Bayesian Optimization iterator from problem description

        Args:
            config (dict): Dictionary with QUEENS problem description

        Returns:
            iterator: BayesOptIterator object

        """
        method_options = config["method"]["method_options"]
        model_name = method_options["model"]
        model = Model.from_config_create_model(model_name, config)
        return cls(model, method_options["seed"], method_options["num_iter"],
                   method_options["use_ard"], method_options["num_initial_samples"])

    def eval_model(self):
        """ Evaluate the model """
        return self.model.evaluate()

    def prep_and_eval_model(self, X):
        """ Make compatible interface """
        self.model.update_model_from_sample_batch(X)
        return self.eval_model()

    def core_run(self):
        """  Run Bayesian Optimization model """
        # Run initial experimental design
        lhd = LatinHyperCube(self.num_initial_samples, self.domain)
        X = lhd.generate()
        Y = self.prep_and_eval_model(X)

        # setup Gaussian process
        model = gpflow.gpr.GPR(X, Y, gpflow.kernels.Matern52(self.num_params,
                                                             ARD=self.use_ard))

        model.kern.lengthscales.transform = gpflow.transforms.Log1pe(1e-3)

        # create the Bayesian optimizer
        alpha = ExpectedImprovement(model)
        optimizer = BayesianOptimizer(self.domain, alpha)

        # Run the Bayesian optimization
        with optimizer.silent():
            self.results = optimizer.optimize(self.prep_and_eval_model,
                                              n_iter=self.num_iter)

    def post_run(self):
        """ Analyze the results """
        print(self.results)
