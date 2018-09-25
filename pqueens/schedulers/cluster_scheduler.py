import pdb
import abc
import re
#import pqueens
import subprocess
import sys
import json

class AbstractClusterScheduler(metaclass=abc.ABCMeta):
    """ Abstract base class for an interface to cluster scheduling software

    Attributes:
        connect_to_resource (list):     Shell commands to connect resource
                                        running the scheduling software
    """

    @abc.abstractmethod
    def submit_command(self, job_name):
        pass

    @abc.abstractmethod
    def output_regexp(self):
        pass

    @abc.abstractmethod
    def get_process_id_from_output(self, output):
        pass

    @abc.abstractmethod
    def alive(self, process_id):
        pass

    def submit(self, job_id, experiment_name, batch, experiment_dir,
               database_address, driver_options):
        """ Function to submit new job to scheduling software on a given resource

        Args:
            job_id (int):               Id of job to submit
            experiment_name (string):   Name of experiment
            batch (string):             Batch number of job
            experiment_dir (string):    Directory of experiment
            database_address (string):  Address of database to connect to
            driver_options (dict):      Options for driver

        Returns:
            int: proccess id of job

        """
        driver_options['experiment_dir'] = experiment_dir
        driver_options['experiment_name'] = experiment_name
        driver_options['job_id'] = job_id
        driver_options['batch'] = batch
        # for now we assume that an ssh tunnel has been set up such
        # that we can connect to the database via localhost:portid
        driver_options['database_address'] = database_address

        # note for posterity getting a dictionary with options properly from here
        # to the driver script is a pain a geetting the quotations right is a
        # nightmare. In any case the stuff below works, so do not touch it

        # convert driver options dict to json
        driver_options_json_str = json.dumps(driver_options)
        # add a little trick to escape quotes in json file
        driver_options_json_str = driver_options_json_str.replace('"',r'"\\""')
        # run it a second time (not quite sure why this is needed, but it
        # does not work without it)
        #driver_options_json_str = json.dumps(driver_options_json_str)
        #driver_options_json_str = "\\'" +driver_options_json_str + "\\'"
        #driver_options_json_str = "'" +driver_options_json_str + "'"
       # driver_options_json_str = driver_options_json_str
       # the '<' is needed for execution of local python scripts on potentially
        # remote servers
        #driver_args = "--export=VAR=" + "'"+ driver_options_json_str + "'"

        #driver_args = r'--export=VAR=\"' + driver_options_json_str + r'\"'

        driver_args = r'\"' + driver_options_json_str + r'\"'

        #driver_args = driver_options_json_str

        # one more time
        # driver_args = json.dumps(driver_args)

       # run_command = ['<', driver_options['driver_file'], driver_args]
        run_command = [driver_options['driver_file'], driver_args]

        # assemble job_name for cluster
        job_name = 'queens_{}_{}'.format(experiment_name, job_id)
        submit_command = self.submit_command(job_name)

        submit_command.extend(run_command)

        command_string = ' '.join(submit_command)

        #pdb.set_trace()
        # Error happens below:ValueError: Invalid file object: <_io.TextIOWrapper name=10 encoding='UTF-8'>
        process = subprocess.Popen(command_string,
                                   stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,
                                   shell=True,
                                   universal_newlines=True)

        output, std_err = process.communicate()
        process.stdin.close()

        # get the process id from text output
        match = self.get_process_id_from_output(output)
        try:
            return int(match)
        except:
            sys.stderr.write(output)
            return None
