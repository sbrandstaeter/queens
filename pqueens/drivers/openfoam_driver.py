import os
import stat
import json
import shutil

from pqueens.drivers.driver import Driver
from pqueens.utils.run_subprocess import run_subprocess
from pqueens.utils.injector import inject
from pqueens.utils.numpy_array_encoder import NumpyArrayEncoder


class OpenFOAMDriver(Driver):
    """ 
    Driver to run OpenFOAM
    """

    def __init__(self, base_settings):
        super(OpenFOAMDriver, self).__init__(base_settings)

    @classmethod
    def from_config_create_driver(cls, base_settings, workdir=None):
        """
        Create Driver to run OpenFOAM from input configuration

        Args:
            base_settings (dict): dictionary with base settings of parent class
                                  (depreciated: will be removed soon)

        Returns:
            OpenFOAMDriver (obj):     instance of OpenFOAMDriver class

        """
        # set destination directory
        dest_dir = os.path.join(str(base_settings['experiment_dir']), str(base_settings['job_id']))

        # extract paths to input directory and two input dictionaries from input
        # template string
        input_dir, input_dic_1, input_dic_2 = base_settings['simulation_input_template'].split()

        # set path to output directory (either on local or remote machine)
        # copy OpenFOAM case directory to output directory
        base_settings['output_directory'] = os.path.join(dest_dir, 'output')

        # make (actual) output directory on remote machine as well as "mirror" output
        # directory on local machine for remote scheduling without Singularity
        if base_settings['remote'] and not base_settings['singularity']:
            # make output directory on remote machine
            copy_directory_to_remote(
                base_settings['remote_connect'], input_dir, base_settings['output_directory']
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
            # copy OpenFOAM case directory to output directory on local machine
            if not os.path.isdir(base_settings['output_directory']):
                shutil.copytree(input_dir, base_settings['output_directory'])

        # set path to case run script and two input files
        base_settings['case_run_script'] = os.path.join(
            base_settings['output_directory'], "run_script"
        )
        base_settings['input_file'] = os.path.join(base_settings['output_directory'], input_dic_1)
        base_settings['input_file_2'] = os.path.join(base_settings['output_directory'], input_dic_2)

        return cls(base_settings)

    # ----------------- CHILD METHODS THAT NEED TO BE IMPLEMENTED -----------------
    def prepare_input_files(self):
        """
        Prepare input files on remote machine in case of remote
        scheduling without Singularity or in all other cases
        """
        # parameters to inject in case run script
        self.inject_params = {'case_directory': self.output_directory}

        if self.remote and not self.singularity:
            self.prepare_input_file_on_remote()
        else:
            # copy OpenFOAM general run script to OpenFOAM case directory as
            # executable/writable case run script, with current case directory
            # inserted
            inject(self.inject_params, self.executable, self.case_run_script)
            os.chmod(self.case_run_script, (stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR))

            # set paths to input files and inject
            inject(self.job['params'], self.input_file, self.input_file)
            inject(self.job['params'], self.input_file_2, self.input_file_2)

    def run_job(self):
        """
        Run OpenFOAM with the following currently available scheduling options:
        
        I) local
        II) remote

        1) standard
        5) ECS task 
        
        """
        # checks
        if self.docker_image is None:
            raise RuntimeError(
                "\nOpenFOAM driver currently only available for OpenFOAM in Docker container!"
            )
        if self.singularity is True:
            raise RuntimeError(
                "\nOpenFOAM driver currently not available in combination with Singularity!"
            )

        # assemble run command for ECS task scheduler (I-5 and II-5)
        if self.scheduler_type == 'ecs_task':
            command_string = self.assemble_ecs_task_run_cmd(self, self.case_run_script)

            # additionally extract ECS task ARN from output and
            # transfer to job database
            self.job['aws_arn'] = extract_string_from_output("taskArn", stdout)

            # set subprocess type 'submit' without stdout/stderr output
            subprocess_type = 'submit'
        elif self.scheduler_type == 'standard':
            # assemble command string for Docker OpenFOAM run
            openfoam_run_cmd = self.assemble_docker_run_cmd(self.case_run_script)

            # assemble remote run command for standard scheduler (II-1)
            if self.remote:
                command_string = self.assemble_remote_run_cmd(openfoam_run_cmd)

            # assemble local run command for standard scheduler (I-1)
            else:
                command_string = openfoam_run_cmd

            # set subprocess type 'simple' with stdout/stderr output
            subprocess_type = 'simple'
        else:
            raise RuntimeError(
                "\nIncorrect scheduler for OpenFOAM driver: currently, only standard and ECS task!"
            )

        # run OpenFOAM via subprocess
        returncode, self.pid, stdout, stderr = run_subprocess(
            command_string, subprocess_type=subprocess_type,
        )

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
        Post-process OpenFOAM job -> currently not required
        """
        pass

    # ----- METHODS FOR PREPARATIVE TASKS ON REMOTE MACHINE ---------------------
    def prepare_input_file_on_remote(self, inject_params):
        """ 
        Prepare input file on remote machine
        """
        # generate two JSON file containing parameter dictionaries
        params_json_name_script = 'params_dict_script.json'
        params_json_name_job = 'params_dict_job.json'
        params_json_path_script = os.path.join(self.global_output_dir, params_json_name_script)
        params_json_path_job = os.path.join(self.global_output_dir, params_json_name_job)
        with open(params_json_path_script, 'w') as fp:
            json.dump(inject_params, fp, cls=NumpyArrayEncoder)
        with open(params_json_path_job, 'w') as fp:
            json.dump(self.job['params'], fp, cls=NumpyArrayEncoder)

        # generate command for copying 'injector.py' and JSON files containing
        # parameter dictionaries to experiment directory on remote machine
        injector_filename = 'injector.py'
        this_dir = os.path.dirname(__file__)
        rel_path = os.path.join('../utils', injector_filename)
        abs_path = os.path.join(this_dir, rel_path)
        command_list = [
            "scp ",
            abs_path,
            " ",
            params_json_path_script,
            " ",
            params_json_path_job,
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
                "\nInjector file and param dict files could not be copied to remote machine!"
                f"\nStderr:\n{stderr}"
            )

        # remove local copies of JSON files containing parameter dictionaries
        os.remove(params_json_path_script)
        os.remove(params_json_path_job)

        # generate command for executing 'injector.py' on remote machine
        injector_path_on_remote = os.path.join(self.experiment_dir, injector_filename)
        script_json_path_on_remote = os.path.join(self.experiment_dir, params_json_name_script)
        job_json_path_on_remote = os.path.join(self.experiment_dir, params_json_name_job)

        arg_list_script = (
            script_json_path_on_remote + ' ' + self.executable + ' ' + self.case_run_script
        )
        arg_list_job_1 = job_json_path_on_remote + ' ' + self.input_file + ' ' + self.input_file
        arg_list_job_2 = job_json_path_on_remote + ' ' + self.input_file_2 + ' ' + self.input_file_2
        command_list = [
            'ssh',
            self.remote_connect,
            '"python',
            injector_path_on_remote,
            arg_list_script,
            '&& python',
            injector_path_on_remote,
            arg_list_job_1,
            '&& python',
            injector_path_on_remote,
            arg_list_job_2,
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

        # generate command for changing mode of run script
        command_list = [
            'ssh',
            self.remote_connect,
            '"chmod 700',
            self.case_run_script,
            '"',
        ]
        command_string = ' '.join(command_list)
        _, _, _, stderr = run_subprocess(command_string)

        # detection of failed command
        if stderr:
            raise RuntimeError(
                "\nMode of run script could not be changed on remote machine!"
                f"\nStderr on remote:\n{stderr}"
            )

        # generate command for removing 'injector.py' and JSON file containing
        # parameter dictionary from experiment directory on remote machine
        command_list = [
            'ssh',
            self.remote_connect,
            '"rm',
            injector_path_on_remote,
            script_json_path_on_remote,
            job_json_path_on_remote,
            '"',
        ]
        command_string = ' '.join(command_list)
        _, _, _, stderr = run_subprocess(command_string)

        # detection of failed command
        if stderr:
            raise RuntimeError(
                "\nInjector and JSON files could not be removed from remote machine!"
                f"\nStderr on remote:\n{stderr}"
            )
