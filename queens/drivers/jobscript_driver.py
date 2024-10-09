"""Driver to run a jobscript."""

import logging
from dataclasses import dataclass
from pathlib import Path

from queens.drivers.driver import Driver
from queens.utils.injector import inject, inject_in_template
from queens.utils.io_utils import read_file
from queens.utils.logger_settings import log_init_args
from queens.utils.metadata import SimulationMetadata
from queens.utils.run_subprocess import run_subprocess_with_logging

_logger = logging.getLogger(__name__)


@dataclass
class JobOptions:
    """Dataclass for job options.

    All the attributes of this dataclass can be injected into input and jobscript files. The input
    files dictionary will be flattened, such that the input files are injected by their key.
    """

    job_dir: Path
    output_dir: Path
    output_file: Path
    job_id: int
    num_procs: int
    experiment_dir: Path
    experiment_name: str
    input_files: dict

    def to_dict(self):
        """Create a job options dict.

        Returns:
            dict: dictionary with all the data
        """
        dictionary = self.__dict__.copy()
        dictionary.update(dictionary.pop("input_files"))
        return dictionary

    def add_data_and_to_dict(self, additional_data):
        """Add additional options to the job options dict.

        Args:
            additional_data (dict): Additional data to combine with the job options

        Returns:
            _type_: _description_
        """
        return self.to_dict() | additional_data


class JobscriptDriver(Driver):
    """Driver to run an executable with a jobscript.

    Attributes:
        input_templates (Path): read in simulation input template as string
        data_processor (obj): instance of data processor class
        gradient_data_processor (obj): instance of data processor class for gradient data
        jobscript_template (str): read in jobscript template as string
        jobscript_options (dict): Dictionary containing jobscript options
        jobscript_file_name (str): Jobscript file name (default: 'jobscript.sh')
    """

    @log_init_args
    def __init__(
        self,
        parameters,
        input_templates,
        jobscript_template,
        executable,
        files_to_copy=None,
        data_processor=None,
        gradient_data_processor=None,
        jobscript_file_name="jobscript.sh",
        extra_options=None,
    ):
        """Initialize JobscriptDriver object.

        Args:
            parameters (Parameters): Parameters object
            input_templates (str, Path, dict): path(s) to simulation input template
            jobscript_template (str, Path): path to jobscript template or read in jobscript template
            executable (str, Path): path to main executable of respective software
            files_to_copy (list, opt): files or directories to copy to experiment_dir
            data_processor (obj, opt): instance of data processor class
            gradient_data_processor (obj, opt): instance of data processor class for gradient data
            jobscript_file_name (str): Jobscript file name (default: 'jobscript.sh')
            extra_options (dict): Extra options to inject into jobscript template
        """
        super().__init__(parameters=parameters, files_to_copy=files_to_copy)
        self.input_templates = self.create_input_templates_dict(input_templates)
        self.files_to_copy.extend(self.input_templates.values())
        self.data_processor = data_processor
        self.gradient_data_processor = gradient_data_processor

        if Path(jobscript_template).is_file():
            self.jobscript_template = read_file(jobscript_template)
        else:
            self.jobscript_template = jobscript_template

        if extra_options is None:
            extra_options = {}

        self.jobscript_options = extra_options
        self.jobscript_options["executable"] = executable
        self.jobscript_file_name = jobscript_file_name

    @staticmethod
    def create_input_templates_dict(input_templates):
        """Cast input templates into a dict.

        Args:
            input_templates (str, Path, dict): Input template(s)

        Returns:
            dict: containing input file names and template paths
        """
        if not isinstance(input_templates, dict):
            input_templates = {"input_file": input_templates}

        input_templates_dict = {
            input_template_key: Path(input_template_path)
            for input_template_key, input_template_path in input_templates.items()
        }
        return input_templates_dict

    def run(self, sample, job_id, num_procs, experiment_dir, experiment_name):
        """Run the driver.

        Args:
            sample (dict): Dict containing sample
            job_id (int): Job ID
            num_procs (int): number of processors
            experiment_name (str): name of QUEENS experiment.
            experiment_dir (Path): Path to QUEENS experiment directory.

        Returns:
            Result and potentially the gradient
        """
        job_dir, output_dir, output_file, input_files, log_file, error_file = self._manage_paths(
            job_id, experiment_dir, experiment_name
        )

        jobscript_file = job_dir / self.jobscript_file_name
        sample_dict = self.parameters.sample_as_dict(sample)

        metadata = SimulationMetadata(job_id=job_id, inputs=sample_dict, job_dir=job_dir)

        with metadata.time_code("prepare_input_files"):
            job_options = JobOptions(
                job_dir=job_dir,
                output_dir=output_dir,
                output_file=output_file,
                job_id=job_id,
                num_procs=num_procs,
                experiment_dir=experiment_dir,
                experiment_name=experiment_name,
                input_files=input_files,
            )

            # Create the input files
            self.prepare_input_files(
                job_options.add_data_and_to_dict(sample_dict), experiment_dir, input_files
            )

            jobscript_file = job_dir / self.jobscript_file_name

            # Create jobscript
            inject_in_template(
                job_options.add_data_and_to_dict(self.jobscript_options),
                self.jobscript_template,
                str(jobscript_file),
            )

        with metadata.time_code("run_jobscript"):
            execute_cmd = "bash " + str(jobscript_file)
            self._run_executable(job_id, execute_cmd, log_file, error_file, verbose=False)

        with metadata.time_code("data_processing"):
            results = self._get_results(output_dir)
            metadata.outputs = results

        return results

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
            input_files (dict): Dict with name and path of the input file(s)
            log_file (Path): Path to log file
            error_file (Path): Path to error file
        """
        job_dir = experiment_dir / str(job_id)
        output_dir = job_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_prefix = experiment_name + "_" + str(job_id)
        output_file = output_dir / output_prefix

        input_files = {}
        for input_template_name, input_template_path in self.input_templates.items():
            input_file_str = (
                f"{experiment_name}_{input_template_name}_{job_id}" + input_template_path.suffix
            )
            input_files[input_template_name] = job_dir / input_file_str

        log_file = output_dir / (output_prefix + ".log")
        error_file = output_dir / (output_prefix + ".err")

        return job_dir, output_dir, output_file, input_files, log_file, error_file

    @staticmethod
    def _run_executable(job_id, execute_cmd, log_file, error_file, verbose=False):
        """Run executable.

        Args:
            job_id (int): Job id
            execute_cmd (str): Executed command
            log_file (Path): Path to log file
            error_file (Path): Path to error file
            verbose (bool, opt): flag for additional streaming to terminal
        """
        run_subprocess_with_logging(
            execute_cmd,
            terminate_expression="PROC.*ERROR",
            logger_name=__name__ + f"_{job_id}",
            log_file=str(log_file),
            error_file=str(error_file),
            streaming=verbose,
            raise_error_on_subprocess_failure=False,
        )

    def _get_results(self, output_dir):
        """Get results from driver run.

        Args:
            output_dir (Path): Path to output directory

        Returns:
            result (np.array): Result from the driver run
            gradient (np.array, None): Gradient from the driver run (potentially None)
        """
        result = None
        if self.data_processor:
            result = self.data_processor.get_data_from_file(output_dir)
            _logger.debug("Got result: %s", result)

        gradient = None
        if self.gradient_data_processor:
            gradient = self.gradient_data_processor.get_data_from_file(output_dir)
            _logger.debug("Got gradient: %s", gradient)
        return result, gradient

    def prepare_input_files(self, sample_dict, experiment_dir, input_files):
        """Prepare and parse data to input files.

        Args:
            sample_dict (dict): Dict containing sample
            experiment_dir (Path): Path to QUEENS experiment directory.
            input_files (dict): Dict with name and path of the input file(s)
        """
        for input_template_name, input_template_path in self.input_templates.items():
            inject(
                sample_dict,
                experiment_dir / input_template_path.name,
                input_files[input_template_name],
            )
