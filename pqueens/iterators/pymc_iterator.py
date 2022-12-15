"""PyMC Iterators base calss."""

import abc
import logging

import aesara.tensor as at
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

from pqueens.iterators.iterator import Iterator
from pqueens.models import from_config_create_model
from pqueens.utils.process_outputs import process_ouputs, write_results
from pqueens.utils.pymc import from_config_create_pymc_distribution_dict

_logger = logging.getLogger(__name__)


class PyMCIterator(Iterator):
    """Iterator based on PyMC.

    References:
        [1]: Salvatier et al. "Probabilistic programming in Python using PyMC3". PeerJ Computer
        Science. 2016.

    Attributes:
        global_settings (dict): Global settings of the QUEENS simulations
        model (obj): Underlying simulation model on which the inverse analysis is conducted
        result_description (dict): Settings for storing and visualizing the results
        discard_tuned_samples (boolean): Setting to discard the samples of the burin-in period
        num_chains (int): Number of chains to sample
        num_burn_in (int):  Number of burn-in steps
        num_samples (int): Number of samples to generate per chain, excluding burn-in period
        num_parameters (int): Actual number of model input parameters that should be calibrated
        chains (np.array): Array with all samples
        seed (int): Seed for the random number generators
        current_sample (np.array): Sample which is currently evaluated
        current_gradient (np.array): Gradient of the currently evaluated sample
        init_strategy (str): Strategy to tune mass damping matrix
        pymc_model (obj): PyMC Model as inference environment
        step (obj): PyMC MCMC method to be used for sampling
        use_queens_prior (boolean): Setting for using the PyMC priors or the QUEENS prior functions
        explicit_distribution_shape (int): Shape of the output, determines number of chains in
        QUEENS
        cores (int): Number of chain to run in parallel
        progressbar (boolean): Setting for printing progress bar while sampling
        logprior (fun): Function to evaluate the QUEENS joint log-prior
        loglike (fun): Function to evaluate QUEENS log-likelihood
        results (obj): PyMC inference object with sampling results
    """

    def __init__(
        self,
        global_settings,
        model,
        num_burn_in,
        num_chains,
        num_samples,
        init_strategy,
        discard_tuned_samples,
        result_description,
        seed,
        use_queens_prior,
        progressbar,
    ):
        """Initialize PyMC iterator.

        Args:
            global_settings (dict): Global settings of the QUEENS simulations
            model (obj): Underlying simulation model on which the inverse analysis is conducted
            num_burn_in (int): Number of burn-in steps
            num_chains (int): Number of chains to sample
            num_samples (int): Number of samples to generate per chain, excluding burn-in period
            init_strategy (str): Strategy to tune mass damping matrix
            discard_tuned_samples (boolean): Setting to discard the samples of the burin-in period
            result_description (dict): Settings for storing and visualizing the results
            seed (int): Seed for rng
            use_queens_prior (boolean): Setting for using the PyMC priors or the QUEENS prior
            functions
            progressbar (boolean): Setting for printing progress bar while sampling
        Returns:
            Initialise pymc iterator
        """
        super().__init__(model, global_settings)
        self.result_description = result_description
        self.discard_tuned_samples = discard_tuned_samples
        self.num_chains = num_chains
        self.num_burn_in = num_burn_in

        if discard_tuned_samples:
            self.num_samples = num_samples
        else:
            self.num_samples = num_samples + num_burn_in

        num_parameters = self.parameters.num_parameters

        self.seed = seed
        np.random.seed(seed)
        self.current_sample = np.zeros((self.num_chains, num_parameters))
        self.current_gradient = np.zeros((self.num_chains, num_parameters))
        self.init_strategy = init_strategy

        self.pymc_model = pm.Model()
        self.step = None
        self.use_queens_prior = use_queens_prior

        self.explicit_distribution_shape = self.num_chains
        self.num_chains = 1
        self.cores = 1

        self.chains = np.zeros((self.explicit_distribution_shape, self.num_samples, num_parameters))

        self.progressbar = progressbar
        if self.use_queens_prior:
            self.logprior = None
        self.loglike = None
        self.results = None
        self.initvals = None

    @staticmethod
    def get_base_attributes_from_config(config, iterator_name, model=None):
        """Create PyMC iterator from problem description.

        Args:
            config (dict): Dictionary with QUEENS problem description
            iterator_name (str): Name of iterator (optional)
            model (model):       Model to use (optional)

        Returns:
            iterator:PyMC Iterator object
        """
        method_options = config[iterator_name]['method_options']
        if model is None:
            model_name = method_options['model']
            model = from_config_create_model(model_name, config)

        result_description = method_options.get('result_description', None)
        global_settings = config.get('global_settings', None)

        num_chains = method_options.get('num_chains', 1)

        num_burn_in = method_options.get('num_burn_in', 0)

        init_strategy = method_options.get('init_strategy', 'auto')
        discard_tuned_samples = method_options.get('discard_tuned_samples', True)

        use_queens_prior = method_options.get('use_queens_prior', False)

        progressbar = method_options.get('progressbar', False)

        return (
            global_settings,
            model,
            num_burn_in,
            num_chains,
            method_options['num_samples'],
            init_strategy,
            discard_tuned_samples,
            result_description,
            method_options['seed'],
            use_queens_prior,
            progressbar,
        )

    def eval_log_prior(self, samples):
        """Evaluate natural logarithm of prior at samples of chains.

        Note: we assume a multiplicative split of prior pdf
        """
        return self.parameters.joint_logpdf(samples).reshape(-1)

    def eval_log_prior_grad(self, samples):
        """Evaluate the gradient of the log-prior.

        Args:
            samples (np.array): Samples to evaluate the gradient at

        Returns:
            (np.array): Gradients
        """
        return self.parameters.grad_joint_logpdf(samples)

    def eval_log_likelihood(self, samples):
        """Evaluate the log-likelihood.

        Args:
             samples (np.array): Samples to evaluate the likelihood at

        Returns:
            (np.array): log-likelihoods
        """
        self.current_sample = samples

        log_likelihood, gradient = self.model.evaluate(samples, gradient_bool=True)

        self.current_gradient = gradient
        return log_likelihood

    def eval_log_likelihood_grad(self, samples):
        """Evaluate the gradient of the log-likelihood.

        Args:
            samples (np.array): Samples to evaluate the gradient at

        Returns:
            (np.array): Gradients
        """
        # pylint: disable-next=fixme
        # TODO: find better way to do this evaluation

        if np.all(self.current_sample == samples):
            gradient = self.current_gradient
        else:
            _, gradient = self.model.evaluate(samples, gradient_bool=True)
        return gradient

    @abc.abstractmethod
    def init_mcmc_method(self):
        """Init the PyMC MCMC Model.

        Args:

        Returns:
            step (obj): The MCMC Method within the PyMC Model
        """

    @abc.abstractmethod
    def init_distribution_wrapper(self):
        """Init the PyMC wrapper for the QUEENS distributions."""

    def pre_run(self):
        """Prepare MCMC run."""
        self.pymc_model.__enter__()
        self.init_distribution_wrapper()
        if self.use_queens_prior:
            _logger.info("Use QUEENS Priors")

            name = self.parameters.names[0]
            prior = pm.DensityDist(
                name,
                logp=self.logprior,
                shape=(self.explicit_distribution_shape, self.parameters.num_parameters),
            )
            initvals_value = self.parameters.draw_samples(self.explicit_distribution_shape)
            self.initvals = {name: initvals_value}
        else:
            _logger.info("Use PyMC Priors")
            prior_list = from_config_create_pymc_distribution_dict(
                self.parameters, self.explicit_distribution_shape
            )
            prior = pm.math.concatenate(prior_list, axis=1)

        prior_tensor = at.as_tensor_variable(prior)
        pm.Potential("likelihood", self.loglike(prior_tensor))
        self.step = self.init_mcmc_method()

    def core_run(self):
        """Core run of PyMC iterator."""
        self.results = pm.sample(
            draws=self.num_samples,
            step=self.step,
            init=self.init_strategy,
            cores=self.cores,
            chains=self.num_chains,
            initvals=self.initvals,
            tune=self.num_burn_in,
            random_seed=self.seed,
            discard_tuned_samples=self.discard_tuned_samples,
            progressbar=self.progressbar,
        )

    def post_run(self):
        """Post-Processing of Results."""
        self.pymc_model.__exit__(None, None, None)
        _logger.info("MCMC by PyMC results:")

        # get the chain as numpy array

        inference_data_dict = {}
        if self.use_queens_prior:
            values = np.swapaxes(
                np.squeeze(self.results.posterior.get(self.parameters.names[0]).to_numpy(), axis=0),
                0,
                1,
            )
            self.chains = values

            inference_data_dict[self.parameters.names[0]] = values
        else:
            current_index = 0
            for num, parameter in enumerate(self.parameters.to_list()):
                values = np.swapaxes(
                    np.squeeze(
                        self.results.posterior.get(self.parameters.names[num]).to_numpy(), axis=0
                    ),
                    0,
                    1,
                )
                self.chains[:, :, current_index : current_index + parameter.dimension] = values
                inference_data_dict[self.parameters.names[num]] = values

                current_index += parameter.dimension

        swaped_chain = np.swapaxes(self.chains, 0, 1)

        # process output takes a dict as input with key 'mean'
        results = process_ouputs(
            {
                'results': self.results,
                'mean': swaped_chain,
            },
            self.result_description,
        )
        if self.result_description["write_results"]:
            write_results(
                results,
                self.global_settings["output_dir"],
                self.global_settings["experiment_name"],
            )

        filebasename = (
            f"{self.global_settings['output_dir']}/{self.global_settings['experiment_name']}"
        )

        idata = az.convert_to_inference_data(inference_data_dict)
        _logger.info("Inference summary:")
        _logger.info(az.summary(idata))

        if self.explicit_distribution_shape > 1 and self.result_description["plot_results"]:
            _logger.info("Generate convergence plots")

            axes = az.plot_trace(idata, combined=True)
            fig = axes.ravel()[0].figure
            fig.savefig(filebasename + "_trace.png")

            axes = az.plot_autocorr(idata)
            fig = axes.ravel()[0].figure
            fig.savefig(filebasename + "_autocorr.png")

            axes = az.plot_forest(idata, combined=True, hdi_prob=0.95, r_hat=True)
            fig = axes.ravel()[0].figure
            fig.savefig(filebasename + "_forest.png")
            plt.close("all")

        _logger.info("MCMC by PyMC results finished")

        return None
