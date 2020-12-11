import os

from pqueens.drivers.driver import Driver
from pqueens.utils.run_subprocess import run_subprocess
from pqueens.utils.injector import inject


class DealIINavierStokesDriver(Driver):
    """
    Driver to run Deal.II Navier-Stokes code
    """

    def __init__(self, base_settings):
        super(DealIINavierStokesDriverDriver, self).__init__(base_settings)

    @classmethod
    def from_config_create_driver(cls, base_settings, workdir=None):
        """
        Create Driver to run Deal.II Navier-Stokes code from input configuration
        and set up required directories and files

        Args:
            base_settings (dict): dictionary with base settings of parent class
                                  (depreciated: will be removed soon)

        Returns:
            DealIINavierStokesDriver (obj): instance of DealIINavierStokesDriver class

        """
        # set destination directory and output prefix
        dest_dir = os.path.join(str(base_settings['experiment_dir']), str(base_settings['job_id']))
        base_settings['output_prefix'] = (
            str(base_settings['experiment_name']) + '_' + str(base_settings['job_id'])
        )

        # set path to input file
        input_file_str = base_settings['output_prefix'] + '.json'
        base_settings['input_file'] = os.path.join(dest_dir, input_file_str)

        # make vtu output directory on local machine, if not already existent
        base_settings['output_directory'] = os.path.join(dest_dir, 'output', 'vtu')
        if not os.path.isdir(base_settings['output_directory']):
            os.makedirs(base_settings['output_directory'])

        # generate path to output file
        base_settings['output_file'] = os.path.join(
            base_settings['output_directory'], base_settings['output_prefix']
        )

        # reset output directory on local machine
        base_settings['output_directory'] = os.path.join(dest_dir, 'output')

        return cls(base_settings)

    # ----------------- CHILD METHODS THAT NEED TO BE IMPLEMENTED -----------------
    def prepare_input_files(self):
        """
        Prepare input file
        """
        inject(self.job['params'], self.simulation_input_template, self.input_file)

        rand_field_realization = self.database.load(
            self.experiment_name, self.batch, 'jobs', {'id': self.job_id}
        )['params']['random_inflow']
        base_path = os.path.dirname(self.simulation_input_template)
        absolute_path = os.path.join(base_path, 'flow_past_cylinder_inflow.txt')
        with open(absolute_path, 'w') as myfile:
            for ele in rand_field_realization:
                myfile.write('%s\n' % ele)

        inject({"output_dir": self.output_directory}, self.input_file, self.input_file)
        inject({"input_dir": absolute_path}, self.input_file, self.input_file)

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
                "\nDeal.II Navier-Stokes driver currently not available with Docker container!"
            )
        if self.singularity:
            raise RuntimeError(
                "\nDeal.II Navier-Stokes driver not available in combination with Singularity!"
            )
        if self.remote:
            raise RuntimeError(
                "\nDeal.II Navier-Stokes driver currently not available for remote scheduling!"
            )

        if self.scheduler_type == 'standard':
            # assemble command string for ANSYS run
            command_string = self.assemble_dealIIns_run_cmd()

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
    def assemble_dealIIns_run_cmd(self):
        """
        Assemble command for Deal.II Navier-Stokes run

        Returns:
            Deal.II Navier-Stokes run command

        """
        mpi_cmd = 'mpirun -np'

        command_list = [
            mpi_cmd,
            str(self.num_procs),
            self.executable,
            self.input_file,
            self.output_file,
        ]

        return ' '.join(filter(None, command_list))
