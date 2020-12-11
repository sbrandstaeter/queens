import os

from pqueens.drivers.driver import Driver
from pqueens.utils.run_subprocess import run_subprocess


class ANSYSDriver(Driver):
    """
    Driver to run ANSYS
    """

    def __init__(self, base_settings):
        super(ANSYSDriver, self).__init__(base_settings)

    @classmethod
    def from_config_create_driver(cls, base_settings, workdir=None):
        """
        Create Driver to run ANSYS from input configuration
        and set up required directories and files

        Args:
            base_settings (dict): dictionary with base settings of parent class
                                  (depreciated: will be removed soon)

        Returns:
            ANSYSDriver (obj):     instance of ANSYSDriver class

        """
        # set destination directory and output prefix
        dest_dir = os.path.join(str(base_settings['experiment_dir']), str(base_settings['job_id']))
        base_settings['output_prefix'] = (
            str(base_settings['experiment_name']) + '_' + str(base_settings['job_id'])
        )

        # set path to input file
        input_file_str = base_settings['output_prefix'] + '.dat'
        base_settings['input_file'] = os.path.join(dest_dir, input_file_str)

        # make output directory on local machine, if not already existent
        base_settings['output_directory'] = os.path.join(dest_dir, 'output')
        if not os.path.isdir(base_settings['output_directory']):
            os.makedirs(base_settings['output_directory'])

        # generate path to output file
        output_file_str = base_settings['output_prefix'] + '.out'
        base_settings['output_file'] = os.path.join(
            base_settings['output_directory'], output_file_str
        )

        return cls(base_settings)

    # ----------------- CHILD METHODS THAT NEED TO BE IMPLEMENTED -----------------
    def prepare_input_files(self):
        """
        Prepare input file
        """
        inject(self.job['params'], self.simulation_input_template, self.input_file)

    def run_job(self):
        """
        Run ANSYS with the following currently available scheduling options:
        
        I) local
        II) remote

        1) standard
        
        """
        # checks
        if self.docker_image is not None:
            raise RuntimeError(
                "\nANSYS driver currently not available for ANSYS in Docker container!"
            )
        if self.singularity:
            raise RuntimeError(
                "\nANSYS driver currently not available in combination with Singularity!"
            )
        if self.remote:
            raise RuntimeError("\nANSYS driver currently not available for remote scheduling!")

        if self.scheduler_type == 'standard':
            # assemble command string for ANSYS run
            command_string = self.assemble_ansys_run_cmd()

            # set subprocess type 'simple' with stdout/stderr output
            subprocess_type = 'simple'
        else:
            raise RuntimeError(
                "\nIncorrect scheduler for ANSYS driver: only standard scheduler available!"
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
        Post-process ANSYS job -> currently not required
        """
        pass

    # ----- COMMAND-ASSEMBLY METHOD ---------------------------------------------
    def assemble_ansys_run_cmd(self):
        """
        Assemble command for ANSYS run

        Returns:
            ANSYS run command

        """
        command_list = []
        if self.ansys_version == 'v15':
            command_list = [
                self.executable,
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
                self.executable,
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
            raise RuntimeError("Unknown ANSYS Version provided in input file!")

        return ' '.join(filter(None, command_list))
