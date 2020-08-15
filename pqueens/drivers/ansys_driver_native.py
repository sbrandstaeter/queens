import os

from pqueens.database.mongodb import MongoDB
from pqueens.drivers.driver import Driver
from pqueens.utils.run_subprocess import run_subprocess


class AnsysDriverNative(Driver):
    """
    Driver to run ANSYS natively on workstation

    Attributes:
        custom_executable (str): Optional custom executable for ANSYS

    Returns:
        AnsysDriverNative_obj (obj): Instance of the AnsysDriverNative instance

    """

    def __init__(self, custom_executable, ansys_version, base_settings):
        # TODO dunder init should not be called with dict
        super(AnsysDriverNative, self).__init__(base_settings)
        self.custom_executable = custom_executable
        self.ansys_version = ansys_version

    @classmethod
    def from_config_create_driver(cls, config, base_settings):
        """
        Create Driver from input file

        Args:
            config (dict):          Input options
            base_settings (dict):   Second dict with input options TODO should probably be removed

        Returns:
            driver: AnsysDriverNative object

        """
        # TODO this needs to be fixed
        database_address = 'localhost:27017'

        database_config = dict(
            global_settings=config["global_settings"],
            database=dict(address=database_address, drop_existing=False),
        )
        db = MongoDB.from_config_create_database(database_config)
        base_settings['database'] = db

        # TODO this is superbad, but the only solution currently
        driver_setting = config["driver"]["driver_params"]

        custom_exec = driver_setting.get('custom_executable', None)
        ansys_version = driver_setting.get('ansys_version', None)
        return cls(custom_exec, ansys_version, base_settings)

    def setup_dirs_and_files(self):
        """ Setup directory structure """

        self.main_executable = self.executable

        # base directories
        dest_dir = os.path.join(str(self.experiment_dir), str(self.job_id))

        self.output_directory = os.path.join(dest_dir, "output")
        if not os.path.isdir(self.output_directory):
            os.makedirs(self.output_directory)

        # create input file name
        input_string = str(self.experiment_name) + '_' + str(self.job_id) + '.dat'
        self.input_file = os.path.join(dest_dir, input_string)

        # create output file name
        output_string = str(self.experiment_name) + '_' + str(self.job_id) + '.out'
        self.output_file = os.path.join(self.output_directory, output_string)

    def run_job(self):
        """ Actual method to run the job on computing machine
            using run_subprocess method from utils
        """
        # assemble run command
        command_string = self.assemble_command_string()

        # run ANSYS via subprocess
        returncode, self.pid, _, _ = run_subprocess(command_string)

        # detection of failed jobs
        if returncode:
            self.result = None
            self.job['status'] = 'failed'

    def assemble_command_string(self):
        """
        Assemble command list

        Returns:
            list (lst): command list to execute ANSYS

        """
        command_list = []
        if self.ansys_version == 'v15':
            command_list = [
                self.main_executable,
                "-b -g -p aa_t_a -dir ",
                self.output_directory,
                "-i ",
                self.input_file,
                "-j ",
                str(self.experiment_name) + '_' + str(self.job_id),
                "-s read -l en-us -t -d X11 > ",
                self.output_file,
            ]
        elif self.ansys_version == 'v19':
            command_list = [
                self.main_executable,
                "-p ansys -smp -np 1 -lch -dir",
                self.output_directory,
                "-j",
                str(self.experiment_name) + '_' + str(self.job_id),
                "-s read -l en-us -b -i",
                self.input_file,
                "-o",
                self.output_file,
            ]
            if self.custom_executable is not None:
                command_list.append("-custom")
                command_list.append(self.custom_executable)
        else:
            raise RuntimeError("Unknown ANSYS Version, fix your config file")

        return ' '.join(filter(None, command_list))
