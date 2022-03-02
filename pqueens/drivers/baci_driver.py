"""Driver for simulation software BACI."""

import json
import logging
import os
import pathlib

import pqueens.database.database as DB_module
from pqueens.drivers.driver import Driver
from pqueens.external_geometry.external_geometry import ExternalGeometry
from pqueens.post_post.post_post import PostPost
from pqueens.randomfields.univariate_field_generator_factory import (
    UniVarRandomFieldGeneratorFactory,
)
from pqueens.utils.cluster_utils import get_cluster_job_id
from pqueens.utils.injector import inject
from pqueens.utils.numpy_array_encoder import NumpyArrayEncoder
from pqueens.utils.run_subprocess import run_subprocess
from pqueens.utils.script_generator import generate_submission_script

_logger = logging.getLogger(__name__)


class BaciDriver(Driver):
    """Driver to run BACI.

    Attributes:
        batch (int):               Current batch of driver calls.
        direct_scheduling(bool):   flag for direct scheduling
        do_postprocessing (str):   string for identifying either local post-processing
                                   ('local') or remote post-processing ('remote') or 'None'
        driver_name (str):         Name of the driver used for the analysis. The name is
                                   specified in the json-input file.
        experiment_dir (str):      path to QUEENS experiment directory
        experiment_name (str):     name of QUEENS experiment
        job (dict):                dictionary containing description of current job
        job_id (int):              job ID as provided in database within range [1, n_jobs]
        num_procs (int):           number of processors for processing
        output_directory (str):    path to output directory (on remote computing resource for
                                   remote scheduling)
        remote (bool):             flag for remote scheduling
        remote connect (str):      (only for remote scheduling) adress of remote
                                   computing resource
        result (np.array):         simulation result to be stored in database
        singularity (bool):        flag for use of Singularity containers
        database (obj):            database object
        cae_output_streaming (bool): flag for additional streaming to given stream
        cluster_options (str):     (only for cluster schedulers Slurm and PBS) cluster options
        log_file (str):            path to log file
        error_file (str):          path to error file
        executable (str):          path to main executable of respective CAE software
        external_geometry_obj (obj): External geometry object
        global_output_dir (str):   path to global output directory provided when launching
        input_file (str):          path to input file
        log_file (str):            path to log file
        num_procs_post (int):      number of processors for post-processing
        output_file (str):         path to output file
        output_prefix (str):       output prefix
        post_file_name_prefix_lst (lst): List with unique prefix sequence to name the
                                         post-processed files by the post-processor
        post_options (list):       (only for post-processing) list containing settings/options
                                   for post-processing
        postprocessor (str):       (only for post-processing) path to postprocessor of
                                   respective CAE software
        random_fields_lst (lst):   List of random fields
        scheduler_type (str):      type of scheduler chosen in QUEENS input file
        workdir (str):             path to working directory
        do_postpostprocessing (bool): Boolean if postpost-processing should be done
        postpostprocessor (obj):   instance of post-post class
        pid (int):                 unique process ID for subprocess
        random_fields_realized_lst (lst): List of random field realizations.
    """

    def __init__(
        self,
        batch,
        direct_scheduling,
        do_postprocessing,
        driver_name,
        experiment_dir,
        experiment_name,
        job,
        job_id,
        num_procs,
        output_directory,
        remote,
        remote_connect,
        result,
        singularity,
        database,
        cae_output_streaming,
        cluster_options,
        control_file,
        error_file,
        executable,
        external_geometry_obj,
        global_output_dir,
        input_file,
        log_file,
        num_procs_post,
        output_file,
        output_prefix,
        post_file_name_prefix_lst,
        post_options,
        postprocessor,
        random_fields_lst,
        scheduler_type,
        simulation_input_template,
        workdir,
        do_postpostprocessing,
        postpostprocessor,
    ):
        """Initialize BaciDriver object.

        Args:
            batch (int):               Current batch of driver calls.
            direct_scheduling(bool):   flag for direct scheduling
            do_postprocessing (str):   string for identifying either local post-processing
                                       ('local') or remote post-processing ('remote') or 'None'
            driver_name (str):         Name of the driver used for the analysis. The name is
                                       specified in the json-input file.
            experiment_dir (str):      path to QUEENS experiment directory
            experiment_name (str):     name of QUEENS experiment
            job (dict):                dictionary containing description of current job
            job_id (int):              job ID as provided in database within range [1, n_jobs]
            num_procs (int):           number of processors for processing
            output_directory (str):    path to output directory (on remote computing resource for
                                       remote scheduling)
            remote (bool):             flag for remote scheduling
            remote_connect (str):      (only for remote scheduling) adress of remote
                                       computing resource
            result (np.array):         simulation result to be stored in database
            singularity (bool):        flag for use of Singularity containers
            database (obj):            database object
            cae_output_streaming (bool): flag for additional streaming to given stream
            cluster_options (str):     (only for cluster schedulers Slurm and PBS) cluster options
            control_file(str):         Baci control file
            log_file (str):            path to log file
            error_file (str):          path to error file
            executable (str):          path to main executable of respective CAE software
            external_geometry_obj (obj): External geometry object
            global_output_dir (str):   path to global output directory provided when launching
            input_file (str):          path to input file
            log_file (str):            path to log file
            num_procs_post (int):      number of processors for post-processing
            output_file (str):         path to output file
            output_prefix (str):       output prefix
            post_file_name_prefix_lst (lst): List with unique prefix sequence to name the
                                             post-processed files by the post-processor
            post_options (list):       (only for post-processing) list containing settings/options
                                       for post-processing
            postprocessor (str):       (only for post-processing) path to postprocessor of
                                       respective CAE software
            random_fields_lst (lst):   List of random fields
            scheduler_type (str):      type of scheduler chosen in QUEENS input file
            simulation_input_template (str): path to BACI input template
            workdir (str):             path to working directory
            do_postpostprocessing (bool): Boolean if postpost-processing should be done
            postpostprocessor (obj):   instance of post-post class
        """
        super().__init__(
            batch,
            direct_scheduling,
            do_postprocessing,
            driver_name,
            experiment_dir,
            experiment_name,
            job,
            job_id,
            num_procs,
            output_directory,
            remote,
            remote_connect,
            result,
            singularity,
            database,
        )
        self.cae_output_streaming = cae_output_streaming
        self.cluster_options = cluster_options
        self.control_file = control_file
        self.error_file = error_file
        self.executable = executable
        self.external_geometry_obj = external_geometry_obj
        self.global_output_dir = global_output_dir
        self.input_file = input_file
        self.log_file = log_file
        self.num_procs_post = num_procs_post
        self.output_file = output_file
        self.output_prefix = output_prefix
        self.pid = None
        self.post_file_name_prefix_lst = post_file_name_prefix_lst
        self.post_options = post_options
        self.postprocessor = postprocessor
        self.random_fields_lst = random_fields_lst
        self.random_fields_realized_lst = []
        self.scheduler_type = scheduler_type
        self.simulation_input_template = simulation_input_template
        self.workdir = workdir
        self.do_postpostprocessing = do_postpostprocessing
        self.postpostprocessor = postpostprocessor

    @classmethod
    def from_config_create_driver(
        cls,
        config,
        job_id,
        batch,
        driver_name,
        workdir=None,
        cluster_options=None,
    ):
        """Create Driver to run BACI from input configuration.

        Set up required directories and files.

        Args:
            config (dict):  Dictionary containing configuration from QUEENS input file
            job_id (int):   Job ID as provided in database within range [1, n_jobs]
            batch (int):    Job batch number (multiple batches possible)
            workdir (str):  Path to working directory on remote resource
            driver_name (str): Name of driver instance that should be realized

        Returns:
            BaciDriver (obj): instance of BaciDriver class
        """
        experiment_name = config['global_settings'].get('experiment_name')
        global_output_dir = config['global_settings'].get('output_dir')

        # If multiple resources are passed an error is raised in the resources module.
        resource_name = list(config['resources'])[0]
        scheduler_name = config['resources'][resource_name]['scheduler']
        scheduler_options = config[scheduler_name]
        scheduler_type = scheduler_options['scheduler_type']
        experiment_dir = scheduler_options['experiment_dir']
        num_procs = scheduler_options.get('num_procs', 1)
        num_procs_post = scheduler_options.get('num_procs_post', 1)
        if scheduler_options.get('remote', False):
            remote = True
            remote_options = scheduler_options['remote']
            remote_connect = remote_options['connect']
        else:
            remote = False
            remote_connect = None
        singularity = scheduler_options.get('singularity', False)

        direct_scheduling = False
        if not singularity:
            if scheduler_type in ['pbs', 'slurm']:
                direct_scheduling = True

        # get cluster options if required
        if not scheduler_type in ['pbs', 'slurm']:
            cluster_options = None

        database = DB_module.database

        driver_options = config[driver_name]['driver_params']
        job = None
        result = None
        simulation_input_template = driver_options.get('input_template', None)
        executable = driver_options['path_to_executable']

        postprocessor = driver_options.get('path_to_postprocessor', None)
        if postprocessor:
            post_file_name_prefix_lst = driver_options.get('post_file_name_prefix_lst', None)
            post_options = driver_options.get('post_process_options', None)
        else:
            post_file_name_prefix_lst = None
            post_options = None

        if postprocessor and ((not direct_scheduling) or post_options):
            if remote and not singularity:
                raise NotImplementedError(
                    "Remote computations without singularity is not implemented"
                )
            elif singularity and (scheduler_type in ['pbs', 'slurm']):
                do_postprocessing = 'cluster_sing'
            else:
                do_postprocessing = 'local'
        else:
            do_postprocessing = None

        do_postpostprocessing = driver_options.get('post_post', None)
        if do_postpostprocessing:
            postpostprocessor = PostPost.from_config_create_post_post(
                config, driver_name=driver_name
            )
            cae_output_streaming = False
        else:
            postpostprocessor = None
            cae_output_streaming = True

        dest_dir = os.path.join(experiment_dir, str(job_id))
        output_directory = os.path.join(dest_dir, 'output')
        output_prefix = experiment_name + '_' + str(job_id)
        output_file = os.path.join(output_directory, output_prefix)
        file_extension_obj = pathlib.PurePosixPath(simulation_input_template)
        input_file_str = output_prefix + file_extension_obj.suffix
        input_file = os.path.join(dest_dir, input_file_str)

        if remote and not singularity:
            raise NotImplementedError(
                "The combination of 'remote: true' and 'singularity: false' in the "
                "'scheduler section' is not implemented! "
                "Abort..."
            )
        else:
            if not os.path.isdir(output_directory):
                os.makedirs(output_directory)

        control_file_str = output_prefix + '.control'
        control_file = os.path.join(output_directory, control_file_str)
        log_file_str = output_prefix + '.log'
        log_file = os.path.join(output_directory, log_file_str)
        error_file_str = output_prefix + '.err'
        error_file = os.path.join(output_directory, error_file_str)

        if config.get('external_geometry', None):
            external_geometry_obj = ExternalGeometry.from_config_create_external_geometry(config)
        else:
            external_geometry_obj = None

        model_name = config['method']['method_options'].get('model')
        parameter_name = config[model_name].get('parameters')

        random_fields_lst = None
        if parameter_name:
            random_fields = config[parameter_name].get("random_fields")
            if random_fields:
                random_fields_lst = [
                    (name, value['external_definition']) for name, value in random_fields.items()
                ]

        return cls(
            batch,
            direct_scheduling,
            do_postprocessing,
            driver_name,
            experiment_dir,
            experiment_name,
            job,
            job_id,
            num_procs,
            output_directory,
            remote,
            remote_connect,
            result,
            singularity,
            database,
            cae_output_streaming,
            cluster_options,
            control_file,
            error_file,
            executable,
            external_geometry_obj,
            global_output_dir,
            input_file,
            log_file,
            num_procs_post,
            output_file,
            output_prefix,
            post_file_name_prefix_lst,
            post_options,
            postprocessor,
            random_fields_lst,
            scheduler_type,
            simulation_input_template,
            workdir,
            do_postpostprocessing,
            postpostprocessor,
        )

    # ----------------- CHILD METHODS THAT NEED TO BE IMPLEMENTED -----------------
    def prepare_input_files(self):
        """Prepare input file on remote machine.

        In case of remote scheduling without Singularity or in all other
        cases.
        """
        inject(self.job['params'], self.simulation_input_template, self.input_file)

        # delete copied file and potential back-up files afterwards to save to space
        if self.external_geometry_obj and ("_copy_" in self.simulation_input_template):
            cmd_lst = [
                "rm -f",
                self.simulation_input_template,
                self.simulation_input_template + '.bak',
            ]
            cmd_str = ' '.join(cmd_lst)
            run_subprocess(cmd_str)

    def run_job(self):
        """Run BACI.

        The following scheduling options exist:

        A) with Singularity containers
        B) without Singularity containers

        I) local
        II) remote

        1) standard
        3) Slurm
        4) PBS
        """
        if self.scheduler_type == 'pbs' or self.scheduler_type == 'slurm':
            returncode = self.run_job_via_script()
        else:
            returncode = self.run_job_via_run_cmd()

        # detection of failed jobs
        if returncode:
            self.result = None
            self.job['status'] = 'failed'
        else:
            # save potential path set above and number of processes to database
            self.job['num_procs'] = self.num_procs
            self.database.save(
                self.job,
                self.experiment_name,
                'jobs_' + self.driver_name,
                str(self.batch),
                {
                    'id': self.job_id,
                    'expt_dir': self.experiment_dir,
                    'expt_name': self.experiment_name,
                },
            )

    def postprocess_job(self):
        """Post-process BACI job."""
        # set output and core of target file opt
        output_file_opt = '--file=' + self.output_file
        target_file_opt_core = '--output=' + self.output_directory

        if self.post_options:
            for option, target_file_prefix in zip(
                self.post_options, self.post_file_name_prefix_lst
            ):
                # set extension of target file opt and final target file opt
                target_file_opt = os.path.join(target_file_opt_core, target_file_prefix)

                # run post-processing command
                self.run_postprocessing_cmd(output_file_opt, target_file_opt, option)

    # ----- RUN METHODS ---------------------------------------------------------
    # overload the parent pre_job_run method
    def pre_job_run(self):
        """Runtime manipulations on the dat-file.

        These are the operation that need to be performed before the actual simulation run. This
        method overloads the same-named parent method.

        Returns:
            None
        """
        if self.external_geometry_obj and self.random_fields_lst:
            # TODO currently we have to perform the main run here a second time
            # TODO this is not optimal and should be changed but ok for now
            self.external_geometry_obj.main_run()
            # realize random field sample form decomposition here
            # pylint: disable=line-too-long
            self.random_fields_realized_lst = (
                UniVarRandomFieldGeneratorFactory.calculate_one_truncated_realization_of_all_fields(
                    self.database,
                    self.job_id,
                    self.experiment_name,
                    self.batch,
                    self.experiment_dir,
                    self.random_fields_lst,
                    self.driver_name,
                )
            )
            # pylint: enable=line-too-long
            self._manipulate_dat_file()
        super(BaciDriver, self).pre_job_run()

    def _manipulate_dat_file(self):
        """Helper method that calls the dat-file manipulation method.

        Only needed if random fields are used.

        Returns:
            None
        """
        # set also new name for copied dat-file
        if self.random_fields_lst:
            self.simulation_input_template = self.external_geometry_obj.write_random_fields_to_dat(
                self.random_fields_realized_lst, self.job_id
            )

    def run_job_via_script(self):
        """Run BACI.

        The following scheduling options exist:

        A) with Singularity containers
        B) without Singularity containers

        I) local
        II) remote

        3) Slurm
        4) PBS
        """
        # assemble run command for Slurm and PBS scheduler with Singularity (A-II-3
        # and A-II-4) and without Singularity (B-I-3, B-I-4, B-II-3, and B-II-4)
        if self.scheduler_type == 'pbs' or self.scheduler_type == 'slurm':
            if self.singularity is True:
                command_string = self.assemble_sing_baci_cluster_job_cmd()
            else:
                command_string = self.assemble_baci_cluster_job_cmd()

                # additionally set path to control file
                self.job['control_file_path'] = self.control_file

        # set subprocess type 'simple' with stdout/stderr output for
        # script-based jobs (CAE stdout/stderr on remote for A-II,
        # submission stdout/stdderr on local for B-II) redirected to
        # log and error file, respectively, and task stdout/stderr for 5
        subprocess_type = 'simple'

        # run BACI via subprocess
        returncode, self.pid, stdout, stderr = run_subprocess(
            command_string, subprocess_type=subprocess_type, raise_error_on_subprocess_failure=False
        )
        # redirect stdout/stderr output to log and error file, respectively,
        # for Slurm, PBS (CAE stdout/stderr on remote for A-II, submission
        # stdout/stderr on local for B-II) and remote standard scheduling
        if self.scheduler_type == 'pbs' or self.scheduler_type == 'slurm':
            if self.singularity is True:
                # pid is handled by ClusterScheduler._submit_singularity
                # stdout is raw baci log
                pass
            else:
                # override the pid with cluster scheduler id
                self.pid = get_cluster_job_id(self.scheduler_type, stdout)

            with open(self.log_file, "a") as text_file:
                print(stdout, file=text_file)
            with open(self.error_file, "a") as text_file:
                print(stderr, file=text_file)

        return returncode

    def run_job_via_run_cmd(self):
        """Run BACI via subprocess.

        The following scheduling options exist:

        A) with Singularity containers
        B) without Singularity containers

        I) local
        II) remote

        1) standard
        """
        # initialize various arguments for subprocess to None and
        # merely change below, if required
        terminate_expr = None
        loggername = None
        log_file = None
        error_file = None

        # assemble core command string for BACI run
        baci_run_cmd = self.assemble_baci_run_cmd()

        # assemble remote run command for standard scheduler (B-II-1)
        if not self.singularity and self.remote:
            command_string = self.assemble_remote_run_cmd(baci_run_cmd)

            # set subprocess type 'simple' with stdout/stderr output
            # redirected to log and error file, respectively, below,
            # which are stored on local machine, for the time being
            subprocess_type = 'simple'

        # assemble local run command for standard scheduler (B-I-1)
        # and run command for Singularity (A)
        else:
            command_string = baci_run_cmd

            # set subprocess type 'simulation' with stdout/stderr output
            subprocess_type = 'simulation'
            terminate_expr = 'PROC.*ERROR'
            loggername = __name__ + f'_{self.job_id}'
            log_file = self.log_file
            error_file = self.error_file

        # run BACI via subprocess
        returncode, self.pid, stdout, stderr = run_subprocess(
            command_string,
            subprocess_type=subprocess_type,
            terminate_expr=terminate_expr,
            loggername=loggername,
            log_file=log_file,
            error_file=error_file,
            streaming=self.cae_output_streaming,
            raise_error_on_subprocess_failure=False,
        )

        # redirect stdout/stderr output to log and error file, respectively,
        # for remote standard scheduling
        if self.scheduler_type == 'standard' and self.remote:
            with open(self.log_file, "a") as text_file:
                print(stdout, file=text_file)
            with open(self.error_file, "a") as text_file:
                print(stderr, file=text_file)

        return returncode

    def run_postprocessing_cmd(self, output_file_opt, target_file_opt, option):
        """Run command for postprocessing."""
        # assemble post-processing command with three options:
        # 1) post-processing with Singularity container on cluster with Slurm or PBS
        # 2) local post-processing
        # 3) remote post-processing
        if self.do_postprocessing == 'cluster_sing':
            final_pp_cmd = self.assemble_sing_postprocessing_cmd(
                output_file_opt, target_file_opt, option
            )
        else:
            pp_cmd = self.assemble_postprocessing_cmd(output_file_opt, target_file_opt, option)

            # wrap up post-processing command for remote scheduling or
            # directly use post-processing command for local scheduling
            if self.do_postprocessing == 'remote':
                final_pp_cmd = self.assemble_remote_postprocessing_cmd(pp_cmd)
            else:
                final_pp_cmd = pp_cmd

        # run post-processing command and print potential error messages
        run_subprocess(
            final_pp_cmd,
            additional_error_message="Post-processing of BACI failed!",
            raise_error_on_subprocess_failure=False,
        )

    # ----- COMMAND-ASSEMBLY METHODS ---------------------------------------------
    def assemble_sing_baci_cluster_job_cmd(self):
        """Assemble command for Slurm- or PBS-based BACI run with Singularity.

        Returns:
            Slurm- or PBS-based BACI run command with Singularity
        """
        command_list = [
            'cd',
            self.workdir,
            r'&&',
            self.executable,
            self.input_file,
            self.output_prefix,
        ]

        return ' '.join(filter(None, command_list))

    def assemble_baci_cluster_job_cmd(self):
        """Assemble command for Slurm- or PBS-based BACI run.

        Returns:
            Slurm- or PBS-based BACI run command
        """
        # set options for jobscript
        self.cluster_options['job_name'] = '{}_{}_{}'.format(
            self.experiment_name, 'queens', self.job_id
        )
        self.cluster_options['DESTDIR'] = self.output_directory
        self.cluster_options['EXE'] = self.executable
        self.cluster_options['INPUT'] = self.input_file
        self.cluster_options['OUTPUTPREFIX'] = self.output_prefix
        if self.postprocessor:
            self.cluster_options['POSTPROCESSFLAG'] = 'true'
            self.cluster_options['POSTEXE'] = self.postprocessor
        else:
            self.cluster_options['POSTPROCESSFLAG'] = 'false'
            self.cluster_options['POSTEXE'] = ''

        # set path for script location
        jobfilename = 'jobfile.sh'
        if not self.remote:
            submission_script_path = os.path.join(self.experiment_dir, jobfilename)
        else:
            submission_script_path = os.path.join(self.global_output_dir, jobfilename)

        # generate job script for submission
        generate_submission_script(
            self.cluster_options, submission_script_path, self.cluster_options['jobscript_template']
        )

        if not self.remote:
            # change directory
            os.chdir(self.experiment_dir)

            # assemble command string for jobscript-based run
            command_list = [self.cluster_options['start_cmd'], submission_script_path]
        else:
            # submit the job with jobfile.sh on remote machine
            command_list = [
                "scp ",
                submission_script_path,
                " ",
                self.remote_connect,
                ":",
                self.experiment_dir,
            ]
            command_string = ''.join(command_list)
            run_subprocess(
                command_string,
                additional_error_message="Jobscript could not be copied to remote machine!",
            )

            # remove local copy of submission script and change path
            # to submission script to the one on remote machine
            os.remove(submission_script_path)
            submission_script_path = os.path.join(self.experiment_dir, jobfilename)

            # submit the job with jobfile.sh on remote machine
            command_list = [
                'ssh',
                self.remote_connect,
                '"cd',
                self.experiment_dir,
                ';',
                self.cluster_options['start_cmd'],
                submission_script_path,
                '"',
            ]

        return ' '.join(filter(None, command_list))

    def assemble_baci_run_cmd(self):
        """Assemble command for BACI run.

        Returns:
            BACI run command
        """
        # set MPI command
        if self.remote:
            mpi_cmd = 'mpirun -np'
        else:
            mpi_cmd = 'mpirun --bind-to none -np'

        command_list = [
            mpi_cmd,
            str(self.num_procs),
            self.executable,
            self.input_file,
            self.output_file,
        ]

        return ' '.join(filter(None, command_list))

    def assemble_postprocessing_cmd(self, output_file_opt, target_file_opt, option):
        """Assemble command for postprocessing.

        Args:
            output_file_opt (str): Path (with name) to the simulation output files without the
                                   file extension
            target_file_opt (str): Path (with name) of the post-processed file without the file
                                   extension
            option (str): Post-processing options for external post-processor

        Returns:
            postprocessing command
        """
        # set MPI command
        if self.remote:
            mpi_cmd = 'mpirun -np'
        else:
            mpi_cmd = 'mpirun --bind-to none -np'

        command_list = [
            mpi_cmd,
            str(self.num_procs_post),
            self.postprocessor,
            output_file_opt,
            option,
            target_file_opt,
        ]

        return ' '.join(filter(None, command_list))

    def assemble_sing_postprocessing_cmd(self, output_file_opt, target_file_opt, option):
        """Assemble command for postprocessing in Singularity container.

        Returns:
            Singularity postprocessing command
        """
        command_list = [
            self.postprocessor,
            output_file_opt,
            option,
            target_file_opt,
        ]

        return ' '.join(filter(None, command_list))

    def assemble_remote_postprocessing_cmd(self, postprocess_cmd):
        """Assemble command for remote postprocessing.

        Returns:
            remote postprocessing command
        """
        command_list = [
            'ssh',
            self.remote_connect,
            '"',
            postprocess_cmd,
            '"',
        ]

        return ' '.join(filter(None, command_list))
