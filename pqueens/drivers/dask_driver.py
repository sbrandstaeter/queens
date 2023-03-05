"""QUEENS driver module base class."""
import abc
import logging

_logger = logging.getLogger(__name__)


class Driver(metaclass=abc.ABCMeta):
    """Abstract base class for drivers in QUEENS.

    Attributes:
    """

    def __init__(
        self,
        data_processor,
        gradient_data_processor,
        simulation_input_suffix,
        simulation_input_template,
    ):
        """Initialize Driver object.

        Args:
        """
        self.data_processor = data_processor
        self.gradient_data_processor = gradient_data_processor
        self.simulation_input_suffix = simulation_input_suffix
        self.simulation_input_template = simulation_input_template

    @abc.abstractmethod
    def run(self, sample_dict, num_procs, num_procs_post, experiment_dir, experiment_name):
        """Abstract method for driver run."""

    def _manage_paths(self, job_id, experiment_dir, experiment_name):
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
        """Run data processors."""
        (output_dir / 'out_dir').write_text(str(output_dir))
        result = self.data_processor.get_data_from_file(str(output_dir))
        _logger.debug("Got result: %s", result)

        gradient = None
        if self.gradient_data_processor:
            gradient = self.gradient_data_processor.get_data_from_file(str(output_dir))
            _logger.debug("Got gradient: %s", gradient)
        return result, gradient
