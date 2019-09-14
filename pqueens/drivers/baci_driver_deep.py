import os
import re
from pqueens.drivers.driver import Driver


class BaciDriverDeep(Driver):
    """ Driver to run BACI on the HPC cluster schmarrn (via PBS/Torque)

    Args:

    Returns:
    """
    def __init__(self, base_settings, workdir):
        super(BaciDriverDeep, self).__init__(base_settings)
        self.workdir = workdir

    @classmethod
    def from_config_create_driver(cls, config, base_settings, workdir):
        """ Create Driver from JSON input file

        Args:

        Returns:
            driver: BaciDriverDeep object
        """
        base_settings['address'] = '129.187.58.20:' + str(base_settings['port'])
        base_settings['experiment_name'] = config['experiment_name']
        return cls(base_settings, workdir)

    def setup_mpi(self, ntasks):
        pass

    def setup_dirs_and_files(self):
        """ Setup directory structure

            Args:
                driver_options (dict): Options dictionary

            Returns:
                str, str, str: simualtion prefix, name of input file, name of output file
        """
        # base directories
        dest_dir = str(self.experiment_dir) + '/' + str(self.job_id)

        # Depending on the input file, directories will be created locally or on a cluster
        output_directory = os.path.join(dest_dir, 'output')
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)

        # create input file name
        self.input_file = dest_dir + '/' + str(self.experiment_name) + \
                                     '_' + str(self.job_id) + '.dat'  # TODO change hard coding of .dat

        # create output file name
        self.output_file = output_directory + '/' + str(self.experiment_name) + \
                                              '_' + str(self.job_id)
        self.output_scratch = self.experiment_name + '_' + str(self.job_id)

    def run_job(self):
        """ Actual method to run the job on computing machine
            using run_subprocess method from base class
        """
        # assemble run command
        command_list = ['cd', self.workdir, r'&&', self.executable, self.input_file, self.output_scratch]
        # Here we call directly the executable inside the container not the jobscript!
        command_string = ' '.join(filter(None, command_list))
        _, stderr, self.pid = self.run_subprocess(command_string)
        if stderr:
            if re.fullmatch(r'/bin/sh: line 0: cd: /scratch/PBS_\d+.master.cluster: No such file or directory\n', stderr):
                pass
            else:
                self.result = None  # This is necessary to detect failed jobs
                self.job['status'] = 'failed'
