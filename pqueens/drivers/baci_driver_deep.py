import os
import logging
from pqueens.utils.run_subprocess import run_subprocess

from pqueens.drivers.driver import Driver


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
        base_settings['address'] = '129.187.58.20:' + str(base_settings['port'])
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

        # configure and initiate logger for baci job
        loggername = __name__ + f'{self.job_id}'
        joblogger = logging.getLogger(loggername)
        fh = logging.FileHandler(self.output_file + "_BACI_stdout.txt", mode='w', delay=False)
        fh.setLevel(logging.INFO)
        fh.terminator = ''
        efh = logging.FileHandler(self.output_file + "_BACI_stderr.txt", mode='w', delay=False)
        efh.setLevel(logging.ERROR)
        efh.terminator = ''
        ff = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(ff)
        efh.setFormatter(ff)
        joblogger.addHandler(fh)
        joblogger.addHandler(efh)
        joblogger.setLevel(logging.INFO)

        # Call BACI
        returncode, self.pid = run_subprocess(
            command_string,
            subprocess_type='simulation',
            loggername=loggername,
            terminate_expr='PROC.*ERROR',
        )

        if returncode:
            self.result = None  # This is necessary to detect failed jobs
            self.job['status'] = 'failed'
