import os
import sys

from pqueens.database.mongodb import MongoDB
from pqueens.drivers.driver import Driver
from pqueens.utils.run_subprocess import run_subprocess


class BaciDriverDeep(Driver):
    """
    Driver to run BACI on the HPC cluster Schmarrn (via PBS/Torque) at LNM.

    Attributes:
        base_settings (dict): dictionary with settings from the parent class
                              (depreciated: this will be removed in the future)
        workdir (path): working directory on the remote (HPC-cluster)

    Returns:
       BaciDriverDeep_obj (obj): Instance of BaciDriverDeep class

    """

    def __init__(self, base_settings, workdir):
        super(BaciDriverDeep, self).__init__(base_settings)
        self.workdir = workdir

    @classmethod
    def from_config_create_driver(cls, config, base_settings, workdir):
        """
        Create Driver from JSON input file description

        Args:
            base_settings (dict): dictionary with settings from the parent class
                                  (depreciated: this will be removed in the future)
            workdir (path): working directory on the remote (HPC-cluster)
            config (dict): dictionary containing settings from the input file

        Returns:
            driver: BaciDriverDeep object
        """

        database_address = '129.187.58.20:' + str(base_settings['port'])
        database_config = dict(
            global_settings=config["global_settings"],
            database=dict(
                address=database_address, drop_all_existing_dbs=False, reset_database=False
            ),
        )
        db = MongoDB.from_config_create_database(database_config)
        base_settings['database'] = db

        base_settings['experiment_name'] = config['experiment_name']
        return cls(base_settings, workdir)

    def setup_dirs_and_files(self):
        """
        Setup directory structure.

        Args:
            driver_options (dict): Options dictionary

        Returns:
            None

        """
        # base directories
        dest_dir = str(self.experiment_dir) + '/' + str(self.job_id)

        # Depending on the input file, directories will be created locally or on a cluster
        output_directory = os.path.join(dest_dir, 'output')
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)

        # create input file name
        self.input_file = (
            dest_dir + '/' + str(self.experiment_name) + '_' + str(self.job_id) + '.dat'
        )  # TODO change hard coding of .dat

        # create output file name
        self.output_file = (
            output_directory + '/' + str(self.experiment_name) + '_' + str(self.job_id)
        )
        self.output_scratch = self.experiment_name + '_' + str(self.job_id)

    def run_job(self):
        """
        Actual method to run the job on computing machine
        using run_subprocess method from utils

        Returns:
            None
        """
        # assemble run command
        command_list = [
            'cd',
            self.workdir,
            r'&&',
            self.executable,
            self.input_file,
            self.output_scratch,
        ]
        # Here we call directly the executable inside the container not the jobscript!
        command_string = ' '.join(filter(None, command_list))

        # Call BACI
        returncode, self.pid, stdout, stderr = run_subprocess(command_string)
        # Print the stderr of BACI call to pbs error file
        sys.stderr.write(stderr)
        # Print the stdout of BACI call to pbs output file
        sys.stdout.write(stdout)
        # Print the stderr of BACI to a separate file in the output directory
        with open(self.output_file + "_BACI_stderr.txt", "a") as text_file:
            print(stderr, file=text_file)
        # Print the stdout of BACI to a separate file in the output directory
        with open(self.output_file + "_BACI_stdout.txt", "a") as text_file:
            print(stdout, file=text_file)

        if returncode:
            self.result = None  # This is necessary to detect failed jobs
            self.job['status'] = 'failed'
