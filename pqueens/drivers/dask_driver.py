"""QUEENS driver module base class."""
import abc
import logging
import pathlib
from pathlib import Path

from pqueens.data_processor import from_config_create_data_processor

_logger = logging.getLogger(__name__)


class Driver(metaclass=abc.ABCMeta):
    """Abstract base class for drivers in QUEENS.

    Attributes:
        simulation_input_suffix (str): suffix of the simulation input file
        simulation_input_template (str): read in simulation input template as string
        data_processor (obj): instance of data processor class
        gradient_data_processor (obj): instance of data processor class for gradient data
    """

    def __init__(
        self,
        simulation_input_template,
        data_processor,
        gradient_data_processor,
    ):
        """Initialize Driver object.

        Args:
            simulation_input_template (str, Path): path to simulation input template
            data_processor (obj): instance of data processor class
            gradient_data_processor (obj): instance of data processor class for gradient data
        """
        self.simulation_input_suffix = pathlib.PurePosixPath(simulation_input_template).suffix
        self.simulation_input_template = Path(simulation_input_template)
        self.data_processor = data_processor
        self.gradient_data_processor = gradient_data_processor

    @classmethod
    def from_config_create_driver(
        cls,
        config,
        driver_name,
    ):
        """Create Driver to run executable.

        Args:
            config (dict): Dictionary containing configuration from QUEENS input file
            driver_name (str): Name of the driver

        Returns:
            MpiDriver (obj): Instance of MpiDriver class
        """
        driver_options = config[driver_name].copy()
        driver_options.pop('type')

        data_processor = None
        data_processor_name = driver_options.pop('data_processor_name', None)
        if data_processor_name:
            data_processor = from_config_create_data_processor(config, data_processor_name)

        gradient_data_processor = None
        gradient_data_processor_name = driver_options.pop('gradient_data_processor_name', None)
        if gradient_data_processor_name:
            gradient_data_processor = from_config_create_data_processor(
                config, gradient_data_processor_name
            )

        return cls(
            data_processor=data_processor,
            gradient_data_processor=gradient_data_processor,
            **driver_options
        )

    @abc.abstractmethod
    def run(self, sample_dict, num_procs, num_procs_post, experiment_dir, experiment_name):
        """Abstract method for driver run.

        Args:
            sample_dict (dict): Dict containing sample and job id
            num_procs (int): number of cores
            num_procs_post (int): number of cores for post-processing
            experiment_name (str): name of QUEENS experiment.
            experiment_dir (Path): Path to QUEENS experiment directory.

        Returns:
            Result and potentially the gradient
        """

    def _manage_paths(self, job_id, experiment_dir, experiment_name):
        """Manage paths for driver run.

        Args:
            job_id (int): Job id.
            experiment_dir (Path): Path to QUEENS experiment directory.
            experiment_name (str): name of QUEENS experiment.

        Returns:
            job_dir (Path): Path to job directory
            output_dir (Path): Path to output directory
            output_file (Path): Path to output file(s)
            input_file (Path): Path to input file
            log_file (Path): Path to log file
            error_file (Path): Path to error file
        """
        job_dir = experiment_dir / str(job_id)
        output_dir = job_dir / 'output'
        output_dir.mkdir(parents=True, exist_ok=True)

        output_prefix = experiment_name + '_' + str(job_id)
        output_file = output_dir.joinpath(output_prefix)

        input_file_str = output_prefix + self.simulation_input_suffix
        input_file = job_dir.joinpath(input_file_str)

        log_file = output_dir.joinpath(output_prefix + '.log')
        error_file = output_dir.joinpath(output_prefix + '.err')

        return job_dir, output_dir, output_file, input_file, log_file, error_file

    def _get_results(self, output_dir):
        """Get results from driver run.

        Args:
            output_dir (Path): Path to output directory

        Returns:
            result (np.array): Result from the driver run
            gradient (np.array, None): Gradient from the driver run (potentially None)
        """
        result = self.data_processor.get_data_from_file(str(output_dir))
        _logger.debug("Got result: %s", result)

        gradient = None
        if self.gradient_data_processor:
            gradient = self.gradient_data_processor.get_data_from_file(str(output_dir))
            _logger.debug("Got gradient: %s", gradient)
        return result, gradient
