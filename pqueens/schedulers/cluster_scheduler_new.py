"""Cluster scheduler for QUEENS runs."""
import atexit
import os
import sys

import numpy as np

from pqueens.utils.cluster_utils import get_cluster_job_id
from pqueens.utils.manage_singularity import SingularityManager
from pqueens.utils.run_subprocess import run_subprocess
from pqueens.utils.script_generator import generate_submission_script

from .scheduler_new import Scheduler


class ClusterScheduler(Scheduler):
    """Cluster scheduler (either based on Slurm or Torque/PBS) for QUEENS.

    Args:
        experiment_name (str):     name of QUEENS experiment
        input_file (str):          path to QUEENS input file
        restart (bool):            flag for restart
        experiment_dir (str):      path to QUEENS experiment directory
        driver_name (str):         Name of the driver that shall be used for job submission
        config (dict):             dictionary containing configuration as provided in
                                   QUEENS input file
        cluster_options (dict):    (only for cluster schedulers Slurm and PBS) further
                                   cluster options
        singularity (bool):        flag for use of Singularity containers
        scheduler_type (str):      type of scheduler chosen in QUEENS input file
        singularity_manager (obj): instance of Singularity-manager class
        remote (bool):             flag for remote scheduling
        remote connect (str):      (only for remote scheduling) adress of remote
                                   computing resource
        port (int):                (only for remote scheduling with Singularity) port of
                                   remote resource for ssh port-forwarding to database
        process_ids (dict): Dict of process-IDs of the submitted process as value with job_ids as
                           keys
    """

    def __init__(
        self,
        experiment_name,
        input_file,
        restart,
        experiment_dir,
        driver_name,
        config,
        cluster_options,
        singularity,
        scheduler_type,
        singularity_manager,
        remote,
        remote_connect,
    ):
        """Init method for the cluster scheduler.

        Args:
            experiment_name (str):     name of QUEENS experiment
            input_file (str):          path to QUEENS input file
            restart (bool):            flag for restart
            experiment_dir (str):      path to QUEENS experiment directory
            driver_name (str):         Name of the driver that shall be used for job submission
            config (dict):             dictionary containing configuration as provided in
                                       QUEENS input file
            cluster_options (dict):    (only for cluster schedulers Slurm and PBS) further
                                       cluster options
            singularity (bool):        flag for use of Singularity containers
            scheduler_type (str):      type of scheduler chosen in QUEENS input file
            singularity_manager (obj): instance of Singularity-manager class
            remote (bool):             flag for remote scheduling
            remote_connect (str):      (only for remote scheduling) adress of remote
                                       computing resource
            port (int):                (only for remote scheduling with Singularity) port of
                                       remote resource for ssh port-forwarding to database
            process_ids (dict): Dict of process-IDs of the submitted process as value with job_ids
                                as keys
        """
        super().__init__(
            experiment_name,
            input_file,
            restart,
            experiment_dir,
            driver_name,
            config,
            cluster_options,
            singularity,
            scheduler_type,
        )
        self.singularity_manager = singularity_manager
        self.remote = remote
        self.port = None
        self.remote_connect = remote_connect

        # Close the ssh ports at when exiting after the queens run
        atexit.register(self.post_run)

    @classmethod
    def from_config_create_scheduler(cls, config, scheduler_name=None, driver_name=None):
        """Create cluster scheduler (Slurm or Torque/PBS) class for QUEENS.

        Args:
            config (dict): QUEENS input dictionary
            scheduler_name (str): Name of the scheduler
            driver_name (str): Name of the driver

        Returns:
            instance of cluster scheduler class
        """
        if not scheduler_name:
            scheduler_name = "scheduler"
        scheduler_options = config[scheduler_name]

        if not scheduler_options.get("remote", False):
            raise NotImplementedError("Standard scheduler can not be used remotely")

        experiment_name = config['global_settings']['experiment_name']
        experiment_dir = scheduler_options['experiment_dir']
        input_file = config["input_file"]
        restart = config.get("restart", False)
        singularity = scheduler_options.get('singularity', False)
        scheduler_type = scheduler_options["scheduler_type"]

        cluster_options = scheduler_option.get("cluster", {})

        if singularity:
            singularity_input_options = scheduler_options['singularity_settings']
            singularity_manager = SingularityManager(
                remote=remote,
                remote_connect=remote_connect,
                singularity_bind=cluster_options['singularity_bind'],
                singularity_path=cluster_options['singularity_path'],
                input_file=input_file,
            )

        cluster_options['job_name'] = None

        num_procs = scheduler_options.get('num_procs', '1')

        # set cluster options required specifically for PBS or Slurm
        if scheduler_type == 'pbs':
            cluster_options['start_cmd'] = 'qsub'

            rel_path = '../utils/jobscript_pbs.sh'

            cluster_options['pbs_queue'] = cluster_options.get('pbs_queue', 'batch')
            ppn = scheduler_options.get('pbs_num_avail_ppn', '16')
            if num_procs <= ppn:
                cluster_options['pbs_nodes'] = '1'
                cluster_options['pbs_ppn'] = str(num_procs)
            else:
                num_nodes = np.ceil(num_procs / ppn)
                if num_procs % num_nodes == 0:
                    cluster_options['pbs_ppn'] = str(num_procs / num_nodes)
                else:
                    raise ValueError(
                        "Number of tasks not evenly distributable, as required for PBS scheduler!"
                    )
        elif scheduler_type == "slurm":
            cluster_options['start_cmd'] = 'sbatch'

            rel_path = '../utils/jobscript_slurm.sh'

            cluster_options['slurm_ntasks'] = str(num_procs)

            # What is slurm exclusive?
            if cluster_options.get('slurm_exclusive', False):
                cluster_options['slurm_exclusive'] = ''
            else:
                cluster_options['slurm_exclusive'] = '#'

            if cluster_options.get('slurm_exclude', False):
                cluster_options['slurm_exclude'] = ''
                cluster_options['slurm_excl_node'] = cluster_options['slurm_excl_node']
            else:
                cluster_options['slurm_exclude'] = '#'
                cluster_options['slurm_excl_node'] = ''
        else:
            raise ValueError("Know cluster scheduler types are pbs or slurm")

        script_dir = os.path.dirname(__file__)  # absolute path to directory of this file
        abs_path = os.path.join(script_dir, rel_path)
        cluster_options['jobscript_template'] = abs_path

        if singularity_input_options:
            singularity_path = singularity_input_options['cluster_path']
            cluster_options['singularity_path'] = singularity_path
            cluster_options['EXE'] = os.path.join(singularity_path, 'singularity_image.sif')

            cluster_options['singularity_bind'] = singularity_input_options['cluster_bind']

            cluster_options['OUTPUTPREFIX'] = ''
            cluster_options['POSTPROCESSFLAG'] = 'false'
            cluster_options['POSTEXE'] = ''
            cluster_options['POSTPOSTPROCESSFLAG'] = 'true'
        else:
            cluster_options['singularity_path'] = None
            cluster_options['singularity_bind'] = None
            cluster_options['POSTPOSTPROCESSFLAG'] = 'false'

        if scheduler_options.get('remote'):
            remote = True
            remote_connect = scheduler_options['remote']['connect']
        else:
            remote = False
            remote_connect = None

        return cls(
            experiment_name,
            input_file,
            restart,
            experiment_dir,
            driver_name,
            config,
            cluster_options,
            singularity,
            scheduler_type,
            singularity_manager,
            remote,
            port,
            remote_connect,
        )

    # ------------------- CHILD METHODS THAT MUST BE IMPLEMENTED ------------------
    def pre_run(self):
        """Pre-run routine for local and remote computing with Singularity.

        Do automated port-forwarding and copying files/folders.
        """
        # pre-run routines required when using Singularity both local and remote
        if self.singularity is True:
            self.singularity_manager.check_singularity_system_vars()
            self.singularity_manager.prepare_singularity_files()

            # pre-run routines required when using Singularity remote only
            if self.remote:
                _, _, hostname, _ = run_subprocess('hostname -i')
                _, _, username, _ = run_subprocess('whoami')
                address_localhost = username.rstrip() + r'@' + hostname.rstrip()

                self.singularity_manager.kill_previous_queens_ssh_remote(username)
                self.singularity_manager.establish_port_forwarding_local(address_localhost)
                self.port = self.singularity_manager.establish_port_forwarding_remote(
                    address_localhost
                )

                self.singularity_manager.copy_temp_json()
                self.singularity_manager.copy_post_post()

    def _submit_singularity(self, job_id, batch, restart):
        """Submit job remotely to Singularity.

        Args:
            job_id (int):    ID of job to submit
            batch (str):     Batch number of job

        Returns:
            int:            process ID
        """
        if self.remote:
            # "normal" submission
            if not restart:
                # set job name as well as paths to input file and
                # destination directory for jobscript
                self.cluster_options['job_name'] = '{}_{}_{}'.format(
                    self.experiment_name, 'queens', job_id
                )
                self.cluster_options[
                    'INPUT'
                ] = f"--job_id={job_id} --batch={batch} --port={self.port} --path_json="
                f"{self.cluster_options['singularity_path']} --driver_name={self.driver_name}"
                f" --workdir"

                self.cluster_options['DESTDIR'] = os.path.join(
                    str(self.experiment_dir), str(job_id), 'output'
                )

                # generate jobscript for submission
                submission_script_path = os.path.join(self.experiment_dir, 'jobfile.sh')
                generate_submission_script(
                    self.cluster_options,
                    submission_script_path,
                    self.cluster_options['jobscript_template'],
                    self.remote_connect,
                )

                # submit subscript remotely
                cmdlist_remote_main = [
                    'ssh',
                    self.remote_connect,
                    '"cd',
                    self.experiment_dir,
                    ';',
                    self.cluster_options['start_cmd'],
                    submission_script_path,
                    '"',
                ]
                cmd_remote_main = ' '.join(cmdlist_remote_main)
                _, _, stdout, _ = run_subprocess(
                    cmd_remote_main,
                    additional_message_error="The file 'remote_main' in remote singularity image "
                    "could not be executed properly!",
                )

                # check matching of job ID
                match = get_cluster_job_id(self.scheduler_type, stdout)

                try:
                    return int(match)
                except ValueError:
                    sys.stderr.write(stdout)
                    return None
            # restart submission
            else:
                self.cluster_options['EXE'] = (
                    self.cluster_options['singularity_path'] + '/singularity_image.sif'
                )
                self.cluster_options[
                    'INPUT'
                ] = '--job_id={} --batch={} --port={} --path_json={} --driver_name={}'.format(
                    job_id,
                    batch,
                    self.port,
                    self.cluster_options['singularity_path'],
                    self.driver_name,
                )
                command_list = [
                    'singularity run',
                    self.cluster_options['EXE'],
                    self.cluster_options['INPUT'],
                    '--post=true',
                ]
                submission_script_path = ' '.join(command_list)
                cmdlist_remote_main = [
                    'ssh',
                    self.remote_connect,
                    '"cd',
                    self.experiment_dir,
                    ';',
                    submission_script_path,
                    '"',
                ]
                cmd_remote_main = ' '.join(cmdlist_remote_main)
                run_subprocess(cmd_remote_main)

                return 0
        else:
            raise ValueError("\nSingularity cannot yet be used locally on computing clusters!")

    # TODO this method needs to be replaced by job_id/scheduler_id check
    #  we can only check here if job was completed but might still be failed though
    def check_job_completion(self, job):
        """Check whether this job has been completed.

        Args:
            job (dict): Job dict.

        Returns:
            completed (bool): If job is completed
            failed (bool): If job failed.
        """
        # initialize completion and failure flags to false
        # (Note that failure is not checked for cluster scheduler
        #  and returned false in any case.)
        completed = True
        failed = False
        job_id = job['id']

        if self.scheduler_type == 'pbs':
            check_cmd = 'qstat'
            check_loc = -2
        elif self.scheduler_type == 'slurm':
            check_cmd = 'squeue --job'
            check_loc = -4
        else:
            raise RuntimeError('Unknown scheduler type! Abort...')

        if self.remote:
            # set check command, check location and delete command for PBS or SLURM
            # generate check command
            command_list = [
                'ssh',
                self.remote_connect,
                '"',
                check_cmd,
                str(self.process_ids[str(job_id)]),
                '"',
            ]
        else:
            # generate check command
            command_list = [
                check_cmd,
                str(self.process_ids[str(job_id)]),
            ]

        command_string = ' '.join(command_list)
        _, _, stdout, _ = run_subprocess(command_string)

        if stdout:
            # split output string
            output = stdout.split()

            # second/fourth to last entry should be job status
            status = output[check_loc]
            if status in ['Q', 'R', 'H', 'S']:
                completed = False

        return completed, failed

    def post_run(self):
        """Post-run routine.

        Only for remote computing with Singularity: close ports.
        """
        if self.remote and self.singularity:
            self.singularity_manager.close_local_port_forwarding()
            self.singularity_manager.close_remote_port(self.port)
            print('All port-forwardings were closed again.')
