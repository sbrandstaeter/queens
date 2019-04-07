from .drivers import Driver
import os
import subprocess
from pqueens.utils.injector import inject

#### some comments that will be deleted later #####
# so far driver options are assambled in cluster_scheduler file which seems to be wrong
# ssh command is fully designed and then executed the driver file with the main function reading the argument specified in the ssh command after the script name (sys.args[1]) -> similar to argsparser!

#----
# here rather than reading the command: build driver from config (copy partly the stuff done in the scheduler!
# this should be here!
# so far the DB was not used in local baci driver --> this should be changed!
class Baci_driver_native(Driver):
    """ Driver to run BACI natively on workstation

        Args:
            job (dict): Dict containing all information to run the simulation

        Returns:
            float: result
    """
    def __init__(self, driver_options, experiment_dir, experiment_name, job_id, batch, database_adress)
        super(Baci_driver_native, self).__init__(global_settings) #TODO this needs to be adjusted

        self.driver_options = driver_options
        self.experiment_dir = experiment_dir
        self.experiment_name = experiment_name
        self.job_id = job_id
        self.batch = batch
        self.database_adress = database_adress

    @classmethod
    def from_config_create_driver(cls, config) # TODO check if we want optional arguments for names here
        """ Create Driver from input file

        Args:
            config (dict): Dictionary with QUEENS problem description
            driver_name (str): Name of the driver to identify right section
                                in options dict (optional)
        Returns:
            driver: Baci_driver_native object

        """
        driver_options = config["driver"]["driver_options"]
        experiment_dir = driver_options['experiment_dir']
        experiment_name = driver_options['experiment_name']
        job_id = driver_options['job_id']
        batch = driver_options['batch']
        database_adress = driver_options['database_address']

    return cls(driver_options, experiment_dir, experiment_name, job_id, batch, database_adress)


    def setup_dirs_and_files(self):
        """ Setup directory structure

            Args:
                driver_options (dict): Options dictionary

            Returns:
                str, str, str: simualtion prefix, name of input file, name of output file
        """
        dest_dir = str(driver_options['experiment_dir']) + '/' + \
                  str(driver_options['job_id'])

        prefix = str(driver_options['experiment_name']) + '_' + \
                 str(driver_options['job_id'])

        output_directory = os.path.join(dest_dir, 'output')
        if not os.path.isdir(output_directory):
            # make complete directory treesr/sbin:/sbin:"
            os.makedirs(output_directory)

        # create input file name
        baci_input_file = dest_dir + '/' + str(driver_options['experiment_name']) + \
                          '_' + str(driver_options['job_id']) + '.dat'

        # create output file name
        baci_output =  output_directory + '/' + str(driver_options['experiment_name']) + \
                          '_' + str(driver_options['job_id'])

        return prefix, baci_input_file, baci_output


    def init_job(self):
        """ Initialize job in database

            Args:
                driver_options (dict): Options dictionary
                db (MongoDB) :         MongoDB object

            Returns:
                dict: Dictionary with job information

        """
        job = db.load(driver_options['experiment_name'], driver_options['batch'], 'jobs',
                      {'id' : driver_options['job_id']})

        start_time = time.time()
        job['start time'] = start_time

        db.save(job, driver_options['experiment_name'], 'jobs', driver_options['batch'],
                {'id' : driver_options['job_id']})

        sys.stderr.write("Job launching after %0.2f seconds in submission.\n"
                         % (start_time-job['submit time']))

       return job



    def run_job(self):
        """ Actual method to run the job on computing machine """
        """ Run BACI via subprocess

        Args:
            baci_cmd (string):       Command to run BACI

        Returns:
            string: terminal output
        """
        # assemble baci run command
        baci_cmd = driver_params['path_to_executable'] + ' ' + baci_input_file + ' ' + baci_output_file

        # run BACI
        temp_out = run_baci(baci_cmd)
        print("Communicate run baci")
        print(temp_out)


        p = subprocess.Popen(baci_cmd,
                             shell=True)
        temp_out = p.communicate()

        return temp_out


    def finsih_job(self):
        """ Change status of job to compleded in database """

        if success:
            sys.stderr.write("Completed successfully in %0.2f seconds. [%s]\n"
                             % (end_time-start_time, result))

            job['result'] = result
            job['status'] = 'complete'
            job['end time'] = end_time

        else:
            sys.stderr.write("Job failed in %0.2f seconds.\n" % (end_time-start_time))

            # Update metadata.
            job['status'] = 'broken'
            job['end time'] = end_time

        db.save(job, experiment_name, 'jobs', batch, {'id' : job_id})



    def do_postprocessing(self):
        """ Assemble post processing command """

        # Post-process BACI run
        for i, post_process_option in enumerate(driver_params['post_process_options']):
            post_cmd = driver_params['path_to_postprocessor'] + ' ' + post_process_option + ' --file='+baci_output_file + ' --output='+baci_output_file+'_'+str(i+1)
            temp_out = run_post_processing(post_cmd)
            print("Communicate post-processing")
            print(temp_out)
            #############################

        """ Run BACI post processor via subprocess

        Args:
            post_cmd (string):       Command to run post processing

        Returns:
            string: terminal output
        """

        p = subprocess.Popen(post_cmd,
                             shell=True)
        temp_out = p.communicate()

        return temp_out


    def do_postpostprocessing(self):
        """ Assemble post post processing command """
        """ Run script to extract results from monitor file

        Args:
            post_post_script (string): name of script to run
            baci_output_file (string): name of file to use

        Returns:
            float: actual simulation result
        """
        # call post post process script to extract result from monitor file
        spec = importlib.util.spec_from_file_location("module.name", post_post_script)
        post_post_proc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(post_post_proc)
        result = post_post_proc.run(baci_output_file)

        sys.stderr.write("Got result %s\n" % (result))

        return result


    def setup_mpi(self): #TODO: check how to do this for localhost
        """ Configure and set up the environment for multi_threats

            Args:
                num_procs (int): Number of processors to use

            Returns:
                str, str: MPI runcommand, MPI flags
        """

        mpi_run = 'mpirun'

        if num_procs%16 == 0:
            mpi_flags = "--mca btl openib,sm,self --mca mpi_paffinity_alone 1"
        else:
            mpi_flags = "--mca btl openib,sm,self"

        return mpi_run, mpi_flags, my_env
