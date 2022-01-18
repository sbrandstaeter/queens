"""Cluster scheduler for QUEENS runs."""
import atexit
import os
import sys

import numpy as np

from pqueens.utils.cluster_utils import get_cluster_job_id
from pqueens.utils.run_subprocess import run_subprocess
from pqueens.utils.script_generator import generate_submission_script

from .scheduler import Scheduler


class ClusterScheduler(Scheduler):
    """Cluster scheduler (either based on Slurm or Torque/PBS) for QUEENS."""

    def __init__(self, base_settings):
        """Init method for the cluster scheduler.

        Args:
            base_settings (dict): dictionary containing settings from base class for
                                  further use and completion in this child class
        """
        super(ClusterScheduler, self).__init__(base_settings)

        # Close the ssh ports at when exiting after the queens run
        atexit.register(self.post_run)

    @classmethod
    def create_scheduler_class(cls, base_settings):
        """Create cluster scheduler (Slurm or Torque/PBS) class for QUEENS.

        Args:
            base_settings (dict): dictionary containing settings from base class for
                                  further use and completion in this child class

        Returns:
            scheduler (obj):      instance of scheduler class
        """
        # get input options for scheduler in general and cluster in
        # particular from base settings
        scheduler_input_options = base_settings['scheduler_input_options']
        cluster_input_options = base_settings['cluster_input_options']
        singularity_input_options = base_settings['singularity_input_options']

        # initialize sub-dictionary for cluster options within base settings
        base_settings['cluster_options'] = {}

        # get general cluster options
        base_settings['cluster_options']['job_name'] = None  # job name assembled later
        base_settings['cluster_options']['walltime'] = cluster_input_options['walltime']
        base_settings['cluster_options']['CLUSTERSCRIPT'] = cluster_input_options['script']

        # set output option (currently not enforced)
        if cluster_input_options.get('output', True):
            base_settings['cluster_options']['output'] = ''
        else:
            base_settings['cluster_options']['output'] = '--output=/dev/null --error=/dev/null'

        # get number of processors for processing and post-processing (default: 1)
        ntasks = int(scheduler_input_options.get('num_procs', '1'))
        base_settings['cluster_options']['nposttasks'] = scheduler_input_options.get(
            'num_procs_post', '1'
        )

        # set cluster options required specifically for PBS or Slurm
        if scheduler_input_options['scheduler_type'] == 'pbs':
            # set PBS start command
            base_settings['cluster_options']['start_cmd'] = 'qsub'

            # set relative path to PBS jobscript template
            rel_path = '../utils/jobscript_pbs.sh'

            # for PBS, get type of batch (default: batch)
            base_settings['cluster_options']['pbs_queue'] = cluster_input_options.get(
                'pbs_queue', 'batch'
            )
            # for PBS, split up number of tasks into number of nodes and
            # processors per node
            navppn = int(scheduler_input_options.get('pbs_num_avail_ppn', '16'))
            if ntasks <= navppn:
                base_settings['cluster_options']['pbs_nodes'] = '1'
                base_settings['cluster_options']['pbs_ppn'] = str(ntasks)
            else:
                num_nodes = np.ceil(ntasks / navppn)
                if ntasks % num_nodes == 0:
                    base_settings['cluster_options']['pbs_ppn'] = str(ntasks / num_nodes)
                else:
                    raise ValueError(
                        "Number of tasks not evenly distributable, as required for PBS scheduler!"
                    )
        else:
            # set Slurm start command
            base_settings['cluster_options']['start_cmd'] = 'sbatch'

            # set relative path to Slurm jobscript template
            rel_path = '../utils/jobscript_slurm.sh'

            # for Slurm, directly set number of tasks for processing
            base_settings['cluster_options']['slurm_ntasks'] = str(ntasks)

            # for Slurm, get exclusivity flag (default: false)
            if cluster_input_options.get('slurm_exclusive', False):
                base_settings['cluster_options']['slurm_exclusive'] = ''
            else:
                base_settings['cluster_options']['slurm_exclusive'] = '#'

            # for Slurm, get node-exclusion flag (default: false) as well
            # as potentially excluded nodes
            if cluster_input_options.get('slurm_exclude', False):
                base_settings['cluster_options']['slurm_exclude'] = ''
                base_settings['cluster_options']['slurm_excl_node'] = cluster_input_options[
                    'slurm_excl_node'
                ]
            else:
                base_settings['cluster_options']['slurm_exclude'] = '#'
                base_settings['cluster_options']['slurm_excl_node'] = ''

        # set absolute path to jobscript template
        script_dir = os.path.dirname(__file__)  # absolute path to directory of this file
        abs_path = os.path.join(script_dir, rel_path)
        base_settings['cluster_options']['jobscript_template'] = abs_path

        # set cluster options required for Singularity
        if singularity_input_options is not None:
            # set path to Singularity container in general and
            # also already as executable for jobscript
            singularity_path = singularity_input_options['cluster_path']
            base_settings['cluster_options']['singularity_path'] = singularity_path
            base_settings['cluster_options']['EXE'] = os.path.join(singularity_path, 'image.sif')

            # set cluster bind for Singularity
            base_settings['cluster_options']['singularity_bind'] = singularity_input_options[
                'cluster_bind'
            ]

            # set further fixed options when using Singularity
            base_settings['cluster_options']['OUTPUTPREFIX'] = ''
            base_settings['cluster_options']['POSTPROCESSFLAG'] = 'false'
            base_settings['cluster_options']['POSTEXE'] = ''
            base_settings['cluster_options']['POSTPOSTPROCESSFLAG'] = 'true'
        else:
            # set further fixed options when not using Singularity
            base_settings['cluster_options']['singularity_path'] = None
            base_settings['cluster_options']['singularity_bind'] = None
            base_settings['cluster_options']['POSTPOSTPROCESSFLAG'] = 'false'

        return cls(base_settings)

    # ------------------- CHILD METHODS THAT MUST BE IMPLEMENTED ------------------
    def pre_run(self):
        """Pre-run routine for local and remote computing with Singularity.

        Do automated port-forwarding and copying files/folders.

        Returns:
            None
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
                # pylint: disable=line-too-long
                self.cluster_options[
                    'INPUT'
                ] = '--job_id={} --batch={} --port={} --path_json={} --driver_name={} --workdir '.format(
                    job_id,
                    batch,
                    self.port,
                    self.cluster_options['singularity_path'],
                    self.driver_name,
                )
                # pylint: enable=line-too-long
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
                _, _, stdout, stderr = run_subprocess(cmd_remote_main)

                # error check
                if stderr:
                    raise RuntimeError(
                        "\nThe file 'remote_main' in remote singularity image "
                        "could not be executed properly!"
                        f"\nStderr from remote:\n{stderr}"
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
                    self.cluster_options['singularity_path'] + '/image.sif'
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
                _, _, _, _ = run_subprocess(cmd_remote_main)

                return 0
        else:
            raise ValueError("\nSingularity cannot yet be used locally on computing clusters!")

    # TODO this method needs to be replaced by job_id/scheduler_id check
    #  we can only check here if job was completed but might still be failed though
    def check_job_completion(self, job):
        """Check whether this job has been completed.

        Returns:
            None
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
        _, _, stdout, stderr = run_subprocess(command_string)

        if stdout:
            # split output string
            output = stdout.split()

            # second/fourth to last entry should be job status
            status = output[check_loc]
            if status in ['Q', 'R', 'H', 'S']:
                completed = False

        return completed, failed

    def post_run(self):
        """Post-run routine for remote computing with Singularity: close ports.

        Returns:
            None
        """
        if self.remote and self.singularity:
            self.singularity_manager.close_local_port_forwarding()
            self.singularity_manager.close_remote_port(self.port)
            print('All port-forwardings were closed again.')
