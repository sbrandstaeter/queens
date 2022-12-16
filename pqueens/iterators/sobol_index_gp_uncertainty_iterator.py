"""Iterator for Sobol indices with GP uncertainty."""
import logging
import multiprocessing as mp
import time

import pqueens.parameters.parameters as parameters_module
from pqueens.iterators.sobol_index_gp_uncertainty.estimator import (
    SobolIndexEstimator,
    SobolIndexEstimatorThirdOrder,
)
from pqueens.iterators.sobol_index_gp_uncertainty.predictor import Predictor
from pqueens.iterators.sobol_index_gp_uncertainty.sampler import Sampler, ThirdOrderSampler
from pqueens.iterators.sobol_index_gp_uncertainty.statistics import (
    StatisticsSecondOrderEstimates,
    StatisticsSobolIndexEstimates,
    StatisticsThirdOrderSobolIndexEstimates,
)
from pqueens.models import from_config_create_model
from pqueens.utils.logger_settings import log_through_print
from pqueens.utils.process_outputs import write_results

from .iterator import Iterator

logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
_logger = logging.getLogger(__name__)


class SobolIndexGPUncertaintyIterator(Iterator):
    """Iterator for Sobol indices with metamodel uncertainty.

    This iterator estimates first- and total-order Sobol indices based on Monte-Carlo integration
    and the use of Gaussian process as surrogate model. Additionally, uncertainty estimates for the
    Sobol index estimates are calculates: total uncertainty and separate uncertainty due to Monte-
    Carlo integration and due to the use of the Gaussian process as a surrogate model. Second-order
    indices can optionally be estimated.

    Alternatively, one specific third-order Sobol index can be estimated for one specific
    combination of three parameter (specified as `third_order_parameters` in the input file).

    The approach is based on
    Le Gratiet, Loic, Claire Cannamela, and Bertrand Iooss. ‘A Bayesian Approach
    for Global Sensitivity Analysis of (Multifidelity) Computer Codes’. SIAM/ASA Journal on
    Uncertainty Quantification 2, no. 1 (1 January 2014): 336–63.
    https://doi.org/10.1137/130926869.

    Further details can be found in
    Wirthl, Barbara, Sebastian Brandstaeter, Jonas Nitzler, Bernhard A. Schrefler, and Wolfgang A.
    Wall. ‘Global Sensitivity Analysis Based on Gaussian-Process Metamodelling for Complex
    Biomechanical Problems’. ArXiv:2202.01503 [Cs], 3 February 2022.
    https://arxiv.org/abs/2202.01503.

    Attributes:
        calculate_second_order (bool): true if second-order indices are calculated
        calculate_third_order (bool): true if third-order indices only are calculated
        index_estimator (SobolIndexEstimator object): estimator object
        num_procs (int): number of processors
        parameter_names (list): list of names of input parameters
        predictor (Predictor object): metamodel predictor object
        result_description (dict): dictionary with desired result description
        results (dict): dictionary for results
        sampler (Sampler object): sampler object
        statistics (list): list of statistics objects
    """

    def __init__(
        self,
        model,
        global_settings,
        result_description,
        num_procs,
        sampler,
        predictor,
        index_estimator,
        parameter_names,
        statistics,
        calculate_second_order,
        calculate_third_order,
    ):
        """Initialize Sobol index iterator with GP uncertainty.

        Args:
            model (Model object): QUEENS model to evaluate
            global_settings (dict): dictionary with global (all) settings of the analysis
            result_description (dict): dictionary with desired result description
            num_procs (int): number of processors
            sampler (Sampler): sampler object
            predictor (Predictor object): metamodel predictor object
            index_estimator (SobolIndexEstimator object): estimator object
            parameter_names (list): list of names of input parameters
            statistics (list): list of statistics objects
            calculate_second_order (bool): true if second-order indices are calculated
            calculate_third_order (bool): true if third-order indices only are calculated
        """
        super().__init__(model, global_settings)
        self.result_description = result_description
        self.num_procs = num_procs
        self.sampler = sampler
        self.predictor = predictor
        self.index_estimator = index_estimator
        self.parameter_names = parameter_names
        self.statistics = statistics
        self.calculate_second_order = calculate_second_order
        self.calculate_third_order = calculate_third_order
        self.results = {}

    @classmethod
    def from_config_create_iterator(cls, config, iterator_name, model=None):
        """Create iterator from problem description.

        Args:
            config (dict):       Dictionary with QUEENS problem description
            iterator_name (str): Name of iterator to identify right section
                                 in options dict (optional)
            model (model):       Model to use (optional)

        Returns:
            iterator: SobolIndexGPUncertaintyIterator object
        """
        method_options = config[iterator_name]
        result_description = method_options.get("result_description", None)

        if model is None:
            model_name = method_options['model_name']
            model = from_config_create_model(model_name, config)

        parameter_names = parameters_module.parameters.names

        calculate_second_order = method_options.get("second_order", False)
        calculate_third_order = method_options.get("third_order", False)

        sampler_method, estimator_method = cls._choose_helpers(calculate_third_order)
        sampler = sampler_method.from_config_create(
            method_options, parameter_names, parameters_module.parameters
        )
        index_estimator = estimator_method.from_config_create(method_options, parameter_names)
        predictor = Predictor.from_config_create(method_options, model.interface)

        statistics = []
        if calculate_third_order:
            statistics.append(
                StatisticsThirdOrderSobolIndexEstimates.from_config_create(
                    method_options, parameter_names
                )
            )
        else:
            statistics.append(
                StatisticsSobolIndexEstimates.from_config_create(method_options, parameter_names)
            )
            if calculate_second_order:
                statistics.append(
                    StatisticsSecondOrderEstimates.from_config_create(
                        method_options, parameter_names
                    )
                )

        num_procs = method_options.get("num_procs", mp.cpu_count() - 2)

        _logger.info('Calculate second-order indices is {}'.format(calculate_second_order))

        return cls(
            model,
            config["global_settings"],
            result_description,
            num_procs,
            sampler,
            predictor,
            index_estimator,
            parameter_names,
            statistics,
            calculate_second_order,
            calculate_third_order,
        )

    def pre_run(self):
        """Pre-run."""
        pass

    def core_run(self):
        """Core-run."""
        self.model.build_approximation()

        self.calculate_index()

    def post_run(self):
        """Post-run."""
        if self.result_description is not None:
            if self.result_description["write_results"]:
                write_results(
                    self.results,
                    self.global_settings["output_dir"],
                    self.global_settings['experiment_name'],
                )

    def calculate_index(self):
        """Calculate Sobol indices.

        Run sensitivity analysis based on

        Le Gratiet, Loic, Claire Cannamela, and Bertrand Iooss. ‘A Bayesian Approach
        for Global Sensitivity Analysis of (Multifidelity) Computer Codes’. SIAM/ASA Journal on
        Uncertainty Quantification 2, no. 1 (1 January 2014): 336–63.
        https://doi.org/10.1137/130926869.
        """
        start_run = time.time()

        # 1. Generate Monte-Carlo samples
        samples = self.sampler.sample()
        # 2. Samples realizations of metamodel at Monte-Carlo samples
        prediction = self.predictor.predict(samples, self.num_procs)
        # 3. Calculate Sobol index estimates (including bootstrap)
        estimates = self.index_estimator.estimate(prediction, self.num_procs)

        # 4. Evaluate statistics
        self.evaluate_statistics(estimates)

        _logger.info(f'Time for full calculation: {time.time() - start_run}')

    def evaluate_statistics(self, estimates):
        """Evaluate statistics of Sobol index estimates.

        Args:
            estimates (dict): dictionary of Sobol index estimates of different order
        """
        if self.calculate_third_order:
            self.results['third_order'] = self.statistics[0].evaluate(estimates['third_order'])
            log_through_print(_logger, self.results['third_order'])
        else:
            _logger.info('First-order Sobol indices:')
            self.results['first_order'] = self.statistics[0].evaluate(estimates['first_order'])
            _logger.info('Total-order Sobol indices:')
            self.results['total_order'] = self.statistics[0].evaluate(estimates['total_order'])

            if self.calculate_second_order:
                _logger.info('Second-order Sobol indices:')
                self.results['second_order'] = self.statistics[1].evaluate(
                    estimates['second_order']
                )

    @classmethod
    def _choose_helpers(cls, calculate_third_order):
        """Choose helper objects.

        Choose helper objects for sampling and Sobol index estimating depending on whether we have
        a normal run or a third-order run.

        Returns:
            sampler (type): class type for sampling
            estimator (type): class type for Sobol index estimation
        """
        if calculate_third_order:
            sampler = ThirdOrderSampler
            estimator = SobolIndexEstimatorThirdOrder
        else:
            sampler = Sampler
            estimator = SobolIndexEstimator

        return sampler, estimator
