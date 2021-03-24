import json
import logging
import os
import pathlib

import numpy as np

from pqueens.drivers.driver import Driver
from pqueens.randomfields.univariate_field_generator_factory import (
    UniVarRandomFieldGeneratorFactory,
)  # TODO we should create a unified interface for rf
from pqueens.utils.cluster_utils import get_cluster_job_id
from pqueens.utils.injector import inject
from pqueens.utils.numpy_array_encoder import NumpyArrayEncoder
from pqueens.utils.remote_operations import make_directory_on_remote
from pqueens.utils.run_subprocess import run_subprocess
from pqueens.utils.script_generator import generate_submission_script
from pqueens.utils.string_extractor_and_checker import extract_string_from_output

_logger = logging.getLogger(__name__)


class BaciDriver(Driver):
    """
    Driver to run BACI

    Attributes:
        workdir (str): Path to the working directory of QUEENS experiment
        external_geometry_obj (obj): Object instance of external external_geometry_obj definition
        random_fields_lst (lst): List of descriptions of random fields
        random_fields_realized_lst (lst): List containing one realization of potentially several
                                          random fields
    """

    def __init__(self, base_settings, workdir, external_geometry_obj, random_fields_lst):
        super(BaciDriver, self).__init__(base_settings)
        self.workdir = workdir
        self.external_geometry_obj = external_geometry_obj
        self.random_fields_lst = random_fields_lst
        self.random_fields_realized_lst = []

    @classmethod
    def from_config_create_driver(cls, base_settings, workdir=None):
        """
        Create Driver to run BACI from input configuration
        and set up required directories and files

        Args:
            base_settings (dict): dictionary with base settings of parent class
                                  (depreciated: will be removed soon)
            workdir (str): path to working directory

        Returns:
            BaciDriver (obj): instance of BaciDriver class

        """
        # potentially create an external external_geometry_obj object for dat-file manipulation
        external_geometry_obj = base_settings["external_geometry_obj"]

        # get list of random field tuples: name, type
        random_fields_lst = base_settings["random_fields_lst"]

        # set destination directory and output prefix
        dest_dir = os.path.join(str(base_settings['experiment_dir']), str(base_settings['job_id']))
        base_settings['output_prefix'] = (
            str(base_settings['experiment_name']) + '_' + str(base_settings['job_id'])
        )

        # set path to input file
        file_extension_obj = pathlib.PurePosixPath(base_settings['simulation_input_template'])
        input_file_str = base_settings['output_prefix'] + file_extension_obj.suffix

        base_settings['input_file'] = os.path.join(dest_dir, input_file_str)

        # set path to output directory (either on local or remote machine)
        base_settings['output_directory'] = os.path.join(dest_dir, 'output')

        # make (actual) output directory on remote machine as well as "mirror" output
        # directory on local machine for remote scheduling without Singularity
        if base_settings['remote'] and not base_settings['singularity']:
            # make output directory on remote machine
            make_directory_on_remote(
                base_settings['remote_connect'], base_settings['output_directory']
            )

            # set path to output directory on local machine
            local_dest_dir = os.path.join(
                str(base_settings['global_output_dir']), str(base_settings['job_id'])
            )
            base_settings['local_output_dir'] = os.path.join(local_dest_dir, 'output')

            # make "mirror" output directory on local machine, if not already existent
            if not os.path.isdir(base_settings['local_output_dir']):
                os.makedirs(base_settings['local_output_dir'])
        else:
            # make output directory on local machine, if not already existent
            if not os.path.isdir(base_settings['output_directory']):
                os.makedirs(base_settings['output_directory'])

        # generate path to output files in general as well as control, log and error file
        base_settings['output_file'] = os.path.join(
            base_settings['output_directory'], base_settings['output_prefix']
        )
        control_file_str = base_settings['output_prefix'] + '.control'
        base_settings['control_file'] = os.path.join(
            base_settings['output_directory'], control_file_str
        )
        log_file_str = base_settings['output_prefix'] + '.log'
        base_settings['log_file'] = os.path.join(base_settings['output_directory'], log_file_str)
        error_file_str = base_settings['output_prefix'] + '.err'
        base_settings['error_file'] = os.path.join(
            base_settings['output_directory'], error_file_str
        )

        return cls(base_settings, workdir, external_geometry_obj, random_fields_lst)

    # ----------------- CHILD METHODS THAT NEED TO BE IMPLEMENTED -----------------
    def prepare_input_files(self):
        """
        Prepare input file on remote machine in case of remote
        scheduling without Singularity or in all other cases
        """
        if self.remote and not self.singularity:
            self.prepare_input_file_on_remote()
        else:
            inject(self.job['params'], self.simulation_input_template, self.input_file)

        # delete copied file and potential back-up files afterwards to save to space
        if self.external_geometry_obj:
            cmd_lst = [
                "rm -f",
                self.simulation_input_template,
                self.simulation_input_template + '.bak',
            ]
            cmd_str = ' '.join(cmd_lst)
            _, _, _, stderr = run_subprocess(cmd_str)

    def run_job(self):
        """
        Run BACI with the following scheduling options overall:
        
        A) with Singularity containers
        B) without Singularity containers

        I) local
        II) remote

        1) standard
        2) nohup (not required in combination with A)
        3) Slurm
        4) PBS
        5) ECS task 
        
        """
        if (
            self.scheduler_type == 'pbs'
            or self.scheduler_type == 'slurm'
            or self.scheduler_type == 'ecs_task'
        ):
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
                'jobs',
                str(self.batch),
                {
                    'id': self.job_id,
                    'expt_dir': self.experiment_dir,
                    'expt_name': self.experiment_name,
                },
            )

    def postprocess_job(self):
        """
        Post-process BACI job
        """

        # set output and core of target file opt
        output_file_opt = '--file=' + self.output_file
        target_file_opt_core = '--output=' + self.output_directory

        if self.post_options:
            for num, (option, target_file_prefix) in enumerate(
                zip(self.post_options, self.post_file_name_prefix_lst)
            ):
                # set extension of target file opt and final target file opt
                target_file_opt = os.path.join(target_file_opt_core, target_file_prefix)

                # run post-processing command
                self.run_postprocessing_cmd(output_file_opt, target_file_opt, option)

    # ----- METHODS FOR PREPARATIVE TASKS ON REMOTE MACHINE ---------------------
    def prepare_input_file_on_remote(self):
        """ 
        Prepare input file on remote machine
        """
        # generate a JSON file containing parameter dictionary
        params_json_name = 'params_dict.json'
        params_json_path = os.path.join(self.global_output_dir, params_json_name)
        with open(params_json_path, 'w') as fp:
            json.dump(self.job['params'], fp, cls=NumpyArrayEncoder)

        # generate command for copying 'injector.py' and JSON file containing
        # parameter dictionary to experiment directory on remote machine
        injector_filename = 'injector.py'
        this_dir = os.path.dirname(__file__)
        rel_path = os.path.join('../utils', injector_filename)
        abs_path = os.path.join(this_dir, rel_path)
        command_list = [
            "scp ",
            abs_path,
            " ",
            params_json_path,
            " ",
            self.remote_connect,
            ":",
            self.experiment_dir,
        ]
        command_string = ''.join(command_list)
        _, _, _, stderr = run_subprocess(command_string)

        # detection of failed command
        if stderr:
            raise RuntimeError(
                "\nInjector file and param dict file could not be copied to remote machine!"
                f"\nStderr:\n{stderr}"
            )

        # remove local copy of JSON file containing parameter dictionary
        os.remove(params_json_path)

        # generate command for executing 'injector.py' on remote machine
        injector_path_on_remote = os.path.join(self.experiment_dir, injector_filename)
        json_path_on_remote = os.path.join(self.experiment_dir, params_json_name)

        arg_list = (
            json_path_on_remote + ' ' + self.simulation_input_template + ' ' + self.input_file
        )
        command_list = [
            'ssh',
            self.remote_connect,
            '"',
            self.remote_python_cmd,
            injector_path_on_remote,
            arg_list,
            '"',
        ]
        command_string = ' '.join(command_list)
        _, _, _, stderr = run_subprocess(command_string)

        # detection of failed command
        if stderr:
            raise RuntimeError(
                "\nInjector file could not be executed on remote machine!"
                f"\nStderr on remote:\n{stderr}"
            )

        # generate command for removing 'injector.py' and JSON file containing
        # parameter dictionary from experiment directory on remote machine
        command_list = [
            'ssh',
            self.remote_connect,
            '"rm',
            injector_path_on_remote,
            json_path_on_remote,
            '"',
        ]
        command_string = ' '.join(command_list)
        _, _, _, stderr = run_subprocess(command_string)

        # detection of failed command
        if stderr:
            raise RuntimeError(
                "\nInjector and JSON file could not be removed from remote machine!"
                f"\nStderr on remote:\n{stderr}"
            )

    # ----- RUN METHODS ---------------------------------------------------------
    # overload the parent pre_job_run method
    def pre_job_run(self):
        """
        Runtime manipulations on the dat-file that need to be performed before the actual
        simulation run. This method overloads the same-named parent method.

        Returns:
            None

        """
        if self.external_geometry_obj is not None:
            # TODO currently we have to perform the main run here a second time
            # TODO this is not optimal and should be changed but ok for now
            self.external_geometry_obj.main_run()
            # realize random field sample form decomposition here
            # pylint: disable=line-too-long
            self.random_fields_realized_lst = UniVarRandomFieldGeneratorFactory.calculate_one_truncated_realization_of_all_fields(
                self.database,
                self.job_id,
                self.experiment_name,
                self.batch,
                self.experiment_dir,
                self.random_fields_lst,
            )
            # pylint: enable=line-too-long
            self._manipulate_dat_file()
        super(BaciDriver, self).pre_job_run()

    def _manipulate_dat_file(self):
        """
        Helper method that calls the dat-file manipulation method from the external_geometry_obj.

        Returns:
            None

        """
        # set also new name for copied dat-file
        if self.random_fields_lst is not None:
            self.simulation_input_template = self.external_geometry_obj.write_random_fields_to_dat(
                self.random_fields_realized_lst, self.job_id
            )

    def run_job_via_script(self):
        """
        Run BACI with the following scheduling options:
        
        A) with Singularity containers
        B) without Singularity containers

        I) local
        II) remote

        3) Slurm
        4) PBS
        5) ECS task 
        
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

        # assemble run command for ECS task scheduler, which is always without
        # Singularity (B-I-5 and B-II-5)
        else:
            # assemble command string for BACI run
            baci_run_cmd = self.assemble_baci_run_cmd()

            # assemble command string for ECS task
            command_string = self.assemble_ecs_task_run_cmd(baci_run_cmd)

        # set subprocess type 'simple' with stdout/stderr output for
        # script-based jobs (CAE stdout/stderr on remote for A-II,
        # submission stdout/stdderr on local for B-II) redirected to
        # log and error file, respectively, and task stdout/stderr for 5
        subprocess_type = 'simple'

        # run BACI via subprocess
        returncode, self.pid, stdout, stderr = run_subprocess(
            command_string, subprocess_type=subprocess_type,
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
        # extract ECS task ARN from output for transfer to job database
        elif self.scheduler_type == 'ecs_task':
            self.job['aws_arn'] = extract_string_from_output("taskArn", stdout)

        return returncode

    def run_job_via_run_cmd(self):
        """
        Run BACI with the following scheduling options:
        
        A) with Singularity containers
        B) without Singularity containers

        I) local
        II) remote

        1) standard
        2) nohup (not required in combination with A)
        
        """
        # initialize various arguments for subprocess to None and
        # merely change below, if required
        terminate_expr = None
        loggername = None
        log_file = None
        error_file = None

        # assemble core command string for BACI run
        core_baci_run_cmd = self.assemble_baci_run_cmd()

        # assemble extended core command string for Docker BACI run
        if self.docker_image is not None:
            baci_run_cmd = self.assemble_docker_run_cmd(core_baci_run_cmd)
        else:
            baci_run_cmd = core_baci_run_cmd

        # assemble (local and remote) run command for nohup scheduler
        # (B-I-2 and B-II-2)
        if self.scheduler_type == 'nohup':
            command_string = self.assemble_nohup_baci_run_cmd(baci_run_cmd)

            # additionally set path to log file
            self.job['log_file_path'] = self.log_file

            # set subprocess type 'submit' without stdout/stderr output,
            # since this output will be processed via the nohup command
            subprocess_type = 'submit'
        else:
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
        """ 
        Run command for postprocessing
        """
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
        _, _, _, stderr = run_subprocess(final_pp_cmd)

        # detection of failed command
        if stderr:
            _logger.error("\nPost-processing of BACI failed!" f"\nStderr:\n{stderr}")

    # ----- COMMAND-ASSEMBLY METHODS ---------------------------------------------
    def assemble_sing_baci_cluster_job_cmd(self):
        """  Assemble command for Slurm- or PBS-based BACI run with Singularity

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
        """  Assemble command for Slurm- or PBS-based BACI run

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
        if self.postprocessor is not None:
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
            _, _, _, stderr = run_subprocess(command_string)

            # detection of failed command
            if stderr:
                raise RuntimeError(
                    "\nJobscript could not be copied to remote machine!" f"\nStderr:\n{stderr}"
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
        """  Assemble command for BACI run

            Returns:
                BACI run command

        """
        # set MPI command
        if self.docker_image is not None:
            mpi_cmd = '/usr/lib64/openmpi/bin/mpirun --allow-run-as-root -np'
        else:
            mpi_cmd = 'mpirun -np'

        command_list = [
            mpi_cmd,
            str(self.num_procs),
            self.executable,
            self.input_file,
            self.output_file,
        ]

        return ' '.join(filter(None, command_list))

    def assemble_nohup_baci_run_cmd(self, baci_run_cmd):
        """  Assemble command for nohup run of BACI

            Returns:
                nohup BACI run command

        """
        # assemble command string for nohup BACI run
        nohup_baci_run_cmd = self.assemble_nohup_run_cmd(
            baci_run_cmd, self.log_file, self.error_file
        )

        if self.remote:
            final_nohup_baci_run_cmd = self.assemble_remote_run_cmd(nohup_baci_run_cmd)
        else:
            final_nohup_baci_run_cmd = nohup_baci_run_cmd

        return final_nohup_baci_run_cmd

    def assemble_postprocessing_cmd(self, output_file_opt, target_file_opt, option):
        """
        Assemble command for postprocessing

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
        if self.docker_image is not None:
            mpi_cmd = '/usr/lib64/openmpi/bin/mpirun --allow-run-as-root -np'
        else:
            mpi_cmd = 'mpirun -np'

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
        """  Assemble command for postprocessing in Singularity container

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
        """  Assemble command for remote postprocessing

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
