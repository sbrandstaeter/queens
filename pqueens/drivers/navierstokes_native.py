""" This should be a docstring """

import os
from pqueens.drivers.driver import Driver
from pqueens.utils.injector import inject


class NavierStokesNative(Driver):
    """ Driver to run BACI natively on workstation

        Args:
            job (dict): Dict containing all information to run the simulation

        Returns:
            float: result
    """
    def __init__(self, base_settings):
        super(NavierStokesNative, self).__init__(base_settings)
        self.mpi_config = {}
        self.output_navierstokes = None  # Will be assigned on runtime

    @classmethod
    def from_config_create_driver(cls, config, base_settings, workdir=None):
        """ Create Driver from input file

        Args:
        Returns:
            driver: BaciDriverNative object

        """
        base_settings['address'] = 'localhost:27017'
        return cls(base_settings)

# ----------------- CHILD METHODS THAT NEED TO BE IMPLEMENTED -----------------
    def setup_dirs_and_files(self):
        """ Setup directory structure

            Args:
                driver_options (dict): Options dictionary

            Returns:
                str, str, str: simualtion prefix, name of input file, name of output file
        """
        # base directories
        dest_dir = os.path.join(self.experiment_dir, str(self.job_id))

        # Depending on the input file, directories will be created locally or on a cluster
        output_directory = os.path.join(dest_dir, 'output', 'vtu')
        self.output_navierstokes = os.path.join(dest_dir, 'output/')
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)

        # create input file name
        self.input_file = dest_dir + '/' + str(self.experiment_name) +\
                                     '_' + str(self.job_id) + '.json'

        # create output file name
        self.output_file = output_directory + '/' + str(self.experiment_name) +\
                                              '_' + str(self.job_id)

        self.write_to_file()

    def write_to_file(self):
        """ This should be a docstring """
        rand_field_realization = self.database.load(self.experiment_name, self.batch, 'jobs', {'id': self.job_id})['params']['random_inflow']
        base_path = os.path.dirname(self.template)
        absolute_path = os.path.join(base_path, 'flow_past_cylinder_inflow.txt')

        with open(absolute_path, 'w') as myfile:
            for ele in rand_field_realization:
                myfile.write('%s\n' % ele)

    def run_job(self):
        """ Actual method to run the job on computing machine
            using run_subprocess method from base class
        """
        # write output directory in input file (this is a special case
        # for the navierstokes solver)
        inject({"output_dir": self.output_navierstokes}, self.input_file, self.input_file)
        base_path = os.path.dirname(self.template)
        absolute_path = os.path.join(base_path, 'flow_past_cylinder_inflow.txt')
        inject({"input_dir": absolute_path}, self.input_file, self.input_file)

        # assemble run command
        self.setup_mpi(self.num_procs)
        command_list = [self.executable,
                        self.input_file]
        command_string = ' '.join(filter(None, command_list))
        stdout, stderr, self.pid = self.run_subprocess(command_string)

        if stderr:
            self.result = None
            self.job['status'] = 'failed'

    def setup_mpi(self, num_procs):  # TODO this is not needed atm
        pass
