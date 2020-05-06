import os
from pqueens.drivers.driver import Driver


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
        base_settings['address'] = 'localhost:27017'
        return cls(base_settings)

    # ----------------- CHILD METHODS THAT NEED TO BE IMPLEMENTED -----------------
    def setup_dirs_and_files(self):
        """
        Setup directory structure

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
        )

        # create output file name
        self.output_file = (
            output_directory + '/' + str(self.experiment_name) + '_' + str(self.job_id)
        )

    def run_job(self):
        """
        Actual method to run the job on computing machine
        using run_subprocess method from base class

        Returns:
            None

        """
        # assemble run command sttring
        command_string = self.assemble_command_string()

        # run BACI via subprocess
        stdout, stderr, self.pid = self.run_subprocess(command_string)

        # print the standard output of the subprocess to file (for debugging)
        with open(self.output_file + "_subprocess_stdout.txt", "w") as text_file:
            print(stdout, file=text_file)
        with open(self.output_file + "_subprocess_stderr.txt", "w") as text_file:
            print(stderr, file=text_file)

        # detection of failed jobs
        if stderr:
            self.result = None
            self.job['status'] = 'failed'

    def assemble_command_string(self):
        """  Assemble BACI run command list

            Returns:
                list: command list to execute BACI

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
