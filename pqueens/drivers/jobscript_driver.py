"""Driver to run a jobscript."""
import logging
from pathlib import Path

from pqueens.drivers.driver import Driver
from pqueens.utils.injector import inject_in_template, read_file
from pqueens.utils.run_subprocess import run_subprocess

_logger = logging.getLogger(__name__)


class JobscriptDriver(Driver):
    """Driver to run an executable with mpi.

    Attributes:
        jobscript_template (str): read in jobscript template as string
        jobscript_options (dict): Dictionary containing jobscript options
        jobscript_file_name (str): Jobscript file name (default: 'jobscript.sh')
    """

    def __init__(
        self,
        input_template,
        path_to_executable,
        dask_jobscript_template,
        cluster_script_path,
        post_file_prefix=None,
        post_process_options='',
        path_to_postprocessor=None,
        data_processor=None,
        gradient_data_processor=None,
        jobscript_file_name='jobscript.sh',
    ):
        """Initialize MpiDriver object.

        Args:
            input_template (str, Path): path to simulation input template
            path_to_executable (str, Path): path to main executable of respective software
            dask_jobscript_template (str, Path): path to (dask specific) jobscript template
            cluster_script_path (str, Path): path to cluster script
            post_file_prefix (str, opt): unique prefix to name the post-processed files
            post_process_options (str, opt): options for post-processing
            path_to_postprocessor (path, opt): path to post_processor
            data_processor (obj, opt): instance of data processor class
            gradient_data_processor (obj, opt): instance of data processor class for gradient data
            jobscript_file_name (str): Jobscript file name (default: 'jobscript.sh')
        """
        super().__init__(
            input_template,
            data_processor,
            gradient_data_processor,
        )
        post_processor = path_to_postprocessor if path_to_postprocessor else None

        jobscript_options = {
            "EXE": path_to_executable,
            "BUILDDIR": Path(path_to_executable).parent,
            "OUTPUTPREFIX": post_file_prefix,
            "POSTPROCESS": str(bool(post_processor)).lower(),
            "POSTEXE": str(post_processor),
            "POSTOPTIONS": post_process_options,
            "CLUSTERSCRIPT": cluster_script_path,
        }

        self.jobscript_template = read_file(dask_jobscript_template)
        self.jobscript_options = jobscript_options
        self.jobscript_file_name = jobscript_file_name

    def run(self, sample_dict, _num_procs, _num_procs_post, experiment_dir, experiment_name):
        """Run the driver.

        Args:
            sample_dict (dict): Dict containing sample and job id
            _num_procs (int): number of cores
            _num_procs_post (int): number of cores for post-processing
            experiment_name (str): name of QUEENS experiment.
            experiment_dir (Path): Path to QUEENS experiment directory.

        Returns:
            Result and potentially the gradient
        """
        job_id = sample_dict.pop('job_id')
        job_dir, output_dir, _, input_file, _, _ = self._manage_paths(
            job_id, experiment_dir, experiment_name
        )
        jobscript_file = job_dir.joinpath(self.jobscript_file_name)

        self.prepare_input_files(sample_dict, experiment_dir, input_file)

        final_jobscript_options = {
            **self.jobscript_options,
            'DESTDIR': output_dir,
            'INPUT': input_file,
            'JOB_ID': job_id,
        }
        inject_in_template(final_jobscript_options, self.jobscript_template, str(jobscript_file))

        execute_cmd = 'bash ' + str(jobscript_file)

        run_subprocess(
            execute_cmd,
            subprocess_type='simple',
            raise_error_on_subprocess_failure=False,
        )

        return self._get_results(output_dir)
