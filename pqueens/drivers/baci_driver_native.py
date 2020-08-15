import os

from pqueens.database.mongodb import MongoDB
from pqueens.drivers.driver import Driver
from pqueens.utils.run_subprocess import run_subprocess
from pqueens.utils.script_generator import generate_submission_script


class BaciDriverNative(Driver):
    """
    Driver to run BACI natively on workstation.

    Returns:
        BaciDriverNative_obj (obj): Instance of BaciDriverNative class

    """

    def __init__(self, base_settings):
        super(BaciDriverNative, self).__init__(base_settings)

    @classmethod
    def from_config_create_driver(cls, config, base_settings, workdir=None):
        """
        Create Driver from input file description

        Args:
            config (dict): Dictionary with input configuration
            base_settings (dict): Dictionary with base settings of the parent class
                                  (depreciated: will be removed soon)
            workdir (str): Path to working directory

        Returns:
            BaciDriverNative_obj (obj): Instance of the BaciDriverNative class

        """
        database_address = 'localhost:27017'
        database_config = dict(
            global_settings=config["global_settings"],
            address=database_address,
            drop_all_existing_dbs=False,
            reset_database=False,
        )
        db = MongoDB.from_config_create_database(database_config)
        base_settings['database'] = db
        return cls(base_settings)

    # ----------------- CHILD METHODS THAT NEED TO BE IMPLEMENTED -----------------
    def setup_dirs_and_files(self):
        """
        Setup directory structure

        Returns:
            None

        """
        # set destination directory and output prefix
        dest_dir = os.path.join(str(self.experiment_dir), str(self.job_id))
        self.output_prefix = str(self.experiment_name) + '_' + str(self.job_id)

        # generate path to input file
        input_file_str = self.output_prefix + '.dat'
        self.input_file = os.path.join(dest_dir, input_file_str)

        # make output directory if not already existent
        self.output_directory = os.path.join(dest_dir, 'output')
        if not os.path.isdir(self.output_directory):
            os.makedirs(self.output_directory)

        # generate path to output files in general as well as control file in particular
        self.output_file = os.path.join(self.output_directory, self.output_prefix)
        control_file_str = self.output_prefix + '.control'
        self.control_file = os.path.join(self.output_directory, control_file_str)

    def run_job(self):
        """
        Run BACI natively either directly or via jobscript.

        Returns:
            None

        """
        # decide whether jobscript-based run or direct run
        if self.scheduler_type == 'local_pbs' or self.scheduler_type == 'local_slurm':
            # set options for jobscript
            script_options = {}
            script_options['job_name'] = '{}_{}_{}'.format(
                self.experiment_name, 'queens', self.job_id
            )
            script_options['ntasks'] = self.num_procs
            script_options['DESTDIR'] = self.output_directory
            script_options['EXE'] = self.executable
            script_options['INPUT'] = self.input_file
            script_options['OUTPUTPREFIX'] = self.output_prefix
            script_options['CLUSTERSCRIPT'] = self.cluster_script
            if (self.postprocessor is not None) and (self.post_options is None):
                script_options['POSTPROCESSFLAG'] = 'true'
                script_options['POSTEXE'] = self.postprocessor
                script_options['nposttasks'] = self.num_procs_post
            else:
                script_options['POSTPROCESSFLAG'] = 'false'
                script_options['POSTEXE'] = ''
                script_options['nposttasks'] = ''

            # determine relative path to script template and start command for scheduler
            if self.scheduler_type == 'local_pbs':
                rel_path = '../utils/jobscript_pbs_local.sh'
                scheduler_start = 'qsub'
            else:
                rel_path = '../utils/jobscript_slurm_local.sh'
                scheduler_start = 'sbatch'

            # set paths for script template and final script location
            submission_script_path = os.path.join(self.experiment_dir, 'jobfile.sh')
            this_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
            submission_script_template = os.path.join(this_dir, rel_path)

            # generate job script for submission
            generate_submission_script(
                script_options, submission_script_path, submission_script_template
            )

            # change directory
            os.chdir(self.experiment_dir)

            # assemble command string for jobscript-based run
            command_list = [scheduler_start, submission_script_path]
            command_string = ' '.join(command_list)

            # submit and run job
            returncode, self.pid, stdout, stderr = run_subprocess(command_string)

            # save path to control file and number of processes to database
            self.job['control_file_path'] = self.control_file
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

        else:
            # assemble command string for direct run
            command_string = self.assemble_direct_run_command_string()

            # run BACI via subprocess
            returncode, self.pid, _, _ = run_subprocess(
                command_string,
                subprocess_type='simulation',
                terminate_expr='PROC.*ERROR',
                loggername=__name__ + f'_{self.job_id}',
                output_file=self.output_file,
            )

        # detection of failed jobs
        if returncode:
            self.result = None
            self.job['status'] = 'failed'

    def assemble_direct_run_command_string(self):
        """  Assemble command for direct run of BACI

            Returns:
                direct run command

        """
        # set MPI command
        mpi_command = 'mpirun -np'

        command_list = [
            mpi_command,
            str(self.num_procs),
            self.executable,
            self.input_file,
            self.output_file,
        ]

        return ' '.join(filter(None, command_list))
