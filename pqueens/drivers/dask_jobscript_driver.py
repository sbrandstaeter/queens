"""Driver to run a jobscript."""

import logging
import pathlib

from pqueens.data_processor import from_config_create_data_processor
from pqueens.drivers.dask_driver import Driver
from pqueens.utils.injector import inject_in_template, read_template
from pqueens.utils.run_subprocess import run_subprocess

_logger = logging.getLogger(__name__)


class JobscriptDriver(Driver):
    """Driver to run an executable with mpi.

    Attributes:
    """

    def __init__(
        self,
        jobscript_template,
        jobscript_options,
        simulation_input_suffix,
        simulation_input_template,
        data_processor,
        gradient_data_processor,
    ):
        """Initialize MpiDriver object.

        Args:
            simulation_input_template (path): path to simulation input template (e.g. dat-file)
            data_processor (obj): instance of data processor class
            gradient_data_processor (obj): instance of data processor class for gradient data
        """
        super().__init__(
            data_processor,
            gradient_data_processor,
            simulation_input_suffix,
            simulation_input_template,
        )
        self.jobscript_template = jobscript_template
        self.jobscript_options = jobscript_options

    @classmethod
    def from_config_create_driver(
        cls,
        config,
        driver_name,
    ):
        """Create Driver to run executable from input configuration.

        Set up required directories and files.

        Args:
            config (dict): Dictionary containing configuration from QUEENS input file
            driver_name (str): Name of driver instance that should be realized

        Returns:
            MpiDriver (obj): Instance of MpiDriver class
        """
        driver_options = config[driver_name]
        simulation_input_suffix = pathlib.PurePosixPath(driver_options['input_template']).suffix
        simulation_input_template = read_template(driver_options['input_template'])
        executable = driver_options['path_to_executable']
        jobscript_template = read_template(driver_options['path_to_jobscript'])
        post_processor_str = driver_options.get('path_to_postprocessor', None)
        if post_processor_str:
            post_processor = pathlib.Path(post_processor_str)
        else:
            post_processor = None

        post_file_prefix = driver_options.get('post_file_prefix', None)
        post_options = driver_options.get('post_process_options', '')

        cluster_script_path = driver_options['cluster_script_path']

        data_processor_name = driver_options.get('data_processor_name', None)
        if data_processor_name:
            data_processor = from_config_create_data_processor(config, data_processor_name)
        else:
            data_processor = None

        gradient_data_processor_name = driver_options.get('gradient_data_processor_name', None)
        if gradient_data_processor_name:
            gradient_data_processor = from_config_create_data_processor(
                config, gradient_data_processor_name
            )
        else:
            gradient_data_processor = None

        jobscript_options = {
            "EXE": executable,
            "OUTPUTPREFIX": post_file_prefix,
            "POSTPROCESS": bool(post_processor_str),
            "POSTEXE": str(post_processor),
            "POSTOPTIONS": post_options,
            "CLUSTERSCRIPT": cluster_script_path,
        }

        return cls(
            jobscript_template=jobscript_template,
            jobscript_options=jobscript_options,
            simulation_input_suffix=simulation_input_suffix,
            simulation_input_template=simulation_input_template,
            data_processor=data_processor,
            gradient_data_processor=gradient_data_processor,
        )

    def run(self, sample_dict, num_procs, num_procs_post, experiment_dir, experiment_name):
        """Run the driver.

        Args:
            sample_dict (dict): Dict containing sample and job id
            num_procs (int): number of cores
            num_procs_post (int): number of cores for post-processing
            experiment_name (str): name of QUEENS experiment.
            experiment_dir (Path): Path to QUEENS experiment directory.

        Returns:
            Result and potentially the gradient
        """
        job_id = sample_dict.pop('job_id')
        job_dir, output_dir, _, input_file, _, _ = self._manage_paths(
            job_id, experiment_dir, experiment_name
        )
        jobscript_file = job_dir.joinpath("jobscript.sh")

        inject_in_template(sample_dict, self.simulation_input_template, str(input_file))

        final_jobscript_options = {
            **self.jobscript_options,
            'DESTDIR': output_dir,
            'INPUT': input_file,
            'JOB_ID': job_id,
        }
        inject_in_template(final_jobscript_options, self.jobscript_template, str(jobscript_file))

        execute_cmd = 'bash ' + str(jobscript_file)
        pathlib.Path('/home/dinkel/execute_cmd').write_text(execute_cmd)
        _logger.debug("Start executable with command:")
        _logger.debug(execute_cmd)
        run_subprocess(
            execute_cmd,
            subprocess_type='simple',
            raise_error_on_subprocess_failure=False,
        )

        return self._get_results(output_dir)
