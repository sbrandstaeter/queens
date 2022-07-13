"""Job interface class."""
import logging
import sys
import time

import numpy as np
import pandas as pd

import pqueens.database.database as DB_module
from pqueens.interfaces.interface import Interface
from pqueens.resources.resource import parse_resources_from_configuration
from pqueens.utils.run_subprocess import run_subprocess
from pqueens.utils.user_input import request_user_input_with_default_and_timeout

this = sys.modules[__name__]
this.restart_flag = None

_logger = logging.getLogger(__name__)


class JobInterface(Interface):
    """Class for mapping input variables to responses.

        The JobInterface class maps input variables to outputs, i.e. responses
        by creating a job which is then submitted to a job manager on some
        local or remote resource, which in turn then actually runs the
        simulation software.

    Attributes:
        interface_name (string):                 name of interface
        resources (dict):                        dictionary with resources
        experiment_name (string):                name of experiment
        db (mongodb):                            mongodb to store results and job info
        polling_time (int):                      how frequently do we check if jobs are done
        output_dir (string):                     directory to write output to
        parameters (dict):                       dictionary with parameters
        time_for_data_copy (float): Time (s) to wait such that copying process of simulation
                                    input file can finish and we do not overload the network
    """

    def __init__(
        self,
        interface_name,
        resources,
        experiment_name,
        db,
        polling_time,
        output_dir,
        restart,
        remote,
        remote_connect,
        scheduler_type,
        direct_scheduling,
        time_for_data_copy,
        driver_name,
    ):
        """Create JobInterface.

        Args:
            interface_name (string):    name of interface
            resources (dict):           dictionary with resources
            experiment_name (string):   name of experiment
            db (mongodb):               mongodb to store results and job info
            polling_time (int):         how frequently do we check if jobs are done
            output_dir (string):        directory to write output to
            restart (bool):             true if restart option is chosen
            remote (bool):              true of remote computation
            remote_connect (str):       connection to computing resource
            scheduler_type (str):       scheduler type
            direct_scheduling (bool):   true if direct scheduling
            time_for_data_copy (float): Time (s) to wait such that copying process of simulation
                                        input file can finish and we do not overload the network
            driver_name (str):          Name of the associated driver for the current interface
            _internal_batch_state (int): Helper attribute to compare batch_number with the internal
                                         batch state to detect changes in the batch number.
        """
        super().__init__(interface_name)
        self.name = interface_name
        self.resources = resources
        self.experiment_name = experiment_name
        self.db = db
        self.polling_time = polling_time
        self.output_dir = output_dir
        self.batch_number = 0
        self.num_pending = None
        self.restart = restart
        self.remote = remote
        self.remote_connect = remote_connect
        self.scheduler_type = scheduler_type
        self.direct_scheduling = direct_scheduling
        self.time_for_data_copy = time_for_data_copy
        self.driver_name = driver_name
        self._internal_batch_state = 0

    @classmethod
    def from_config_create_interface(cls, interface_name, config, driver_name):
        """Create JobInterface from config dictionary.

        Args:
            interface_name (str):   name of interface
            config (dict):          dictionary containing problem description
            driver_name (str): Name of the driver that uses this interface

        Returns:
            interface:              instance of JobInterface
        """
        # get experiment name and polling time
        experiment_name = config['global_settings']['experiment_name']
        polling_time = config.get('polling-time', 1.0)

        # get resources from config
        resources = parse_resources_from_configuration(config, driver_name)

        # get various scheduler options
        # TODO: This is not nice
        first = list(config['resources'])[0]
        scheduler_name = config['resources'][first]['scheduler']
        scheduler_options = config[scheduler_name]
        output_dir = scheduler_options["experiment_dir"]
        scheduler_type = scheduler_options['scheduler_type']

        # get flag for remote scheduling
        if scheduler_options.get('remote'):
            remote = True
            remote_connect = scheduler_options['remote']['connect']
        else:
            remote = False
            remote_connect = None

        # get flag for Singularity
        if scheduler_options.get('singularity', False):
            singularity = True
        else:
            singularity = False

        # set flag for direct scheduling
        direct_scheduling = False
        if not singularity:
            if (
                scheduler_type == 'pbs'
                or scheduler_type == 'slurm'
                or (scheduler_type == 'standard' and remote)
            ):
                direct_scheduling = True

        # get flag for restart
        restart = config.get('restart', False)

        db = DB_module.database

        # get waiting time for copying data
        interface_options = config[interface_name]
        time_for_data_copy = interface_options.get('time_for_data_copy')

        # instantiate object
        return cls(
            interface_name,
            resources,
            experiment_name,
            db,
            polling_time,
            output_dir,
            restart,
            remote,
            remote_connect,
            scheduler_type,
            direct_scheduling,
            time_for_data_copy,
            driver_name,
        )

    def evaluate(self, samples, gradient_bool=False):
        """Orchestrate call to external simulation software.

        Second variant which takes the input samples as argument

        Args:
            samples (np.ndarray): realization/samples of QUEENS simulation input variables
            gradient_bool (bool): Flag to determine, whether the gradient of the function at
                                  the evaluation point is expected (True) or not (False)

        Returns:
            output(dict): output data
        """
        self.batch_number += 1

        # Main run
        job_manager = self.get_job_manager()
        jobid_for_data_processor = job_manager(samples)

        # Post run
        for _, resource in self.resources.items():
            if self.direct_scheduling and jobid_for_data_processor.size != 0:
                # check tasks to determine completed jobs
                while not self.all_jobs_finished():
                    time.sleep(self.polling_time)
                    self._check_job_completions(jobid_for_data_processor)

                # submit data processor jobs
                self._manage_data_processor_submission(jobid_for_data_processor)

            # for all other resources:
            else:
                # just wait for all jobs to finish
                while not self.all_jobs_finished():
                    time.sleep(self.polling_time)

        # get sample and response data
        output = self.get_output_data(num_samples=samples.shape[0], gradient_bool=gradient_bool)
        return output

    def get_job_manager(self):
        """Decide whether or not restart is performed.

        Returns:
            function object:    management function which should be used
        """
        if self.restart:
            return self._manage_restart

        else:
            return self._manage_jobs

    def attempt_dispatch(self, resource, new_job):
        """Attempt to dispatch job multiple times.

        This is the actual job submission.
        Submitting jobs to the queue sometimes fails, hence we try multiple times
        before giving up. We also wait one second between submit commands

        Args:
            resource (resource object): Resource to submit job to
            new_job (dict):             Dictionary with job

        Returns:
            int: Process ID of submitted job if successful, None otherwise
        """
        process_id = None
        num_tries = 0

        while process_id is None and num_tries < 5:
            if num_tries > 0:
                time.sleep(0.5)

            # Submit the job to the appropriate resource
            process_id = resource.attempt_dispatch(self.batch_number, new_job)
            num_tries += 1

        return process_id

    def count_jobs(self, field_filters={}):
        """Count jobs matching field_filters in the database.

        default: count all jobs in the database
        Args:
            field_filters: (dict) criteria that jobs to count have to fulfill
        Returns:
            int : number of jobs matching field_filters in the database
        """
        total_num_jobs = 0
        for batch_num in range(1, self.batch_number + 1):
            num_jobs_in_batch = self.db.count_documents(
                self.experiment_name, str(batch_num), 'jobs_' + self.driver_name, field_filters
            )
            total_num_jobs += num_jobs_in_batch

        return total_num_jobs

    def load_jobs(self, field_filters={}):
        """Load jobs that match field_filters from the jobs database.

        Returns:
            list : list with all jobs that match the criteria
        """
        jobs = []
        for batch_num in range(1, self.batch_number + 1):
            job = self.db.load(
                self.experiment_name, str(batch_num), 'jobs_' + self.driver_name, field_filters
            )
            if isinstance(job, list):
                jobs.extend(job)
            else:
                if job is not None:
                    jobs.append(job)

        return jobs

    def save_job(self, job):
        """Save a job to the job database.

        Args:
            job (dict): dictionary with job details
        """
        self.db.save(
            job,
            self.experiment_name,
            'jobs_' + self.driver_name,
            str(self.batch_number),
            {'id': job['id'], 'expt_dir': self.output_dir, 'expt_name': self.experiment_name},
        )

    def create_new_job(self, variables, resource_name, new_id=None):
        """Create new job and save it to database and return it.

        Args:
            variables (Variables):     variables to run model at
            resource_name (string):     name of resource
            new_id (int):                  id for job

        Returns:
            job: new job
        """
        if new_id is None:
            print("Created new job")
            num_jobs = self.count_jobs()
            job_id = num_jobs + 1
        else:
            job_id = int(new_id)

        job = {
            'id': job_id,
            'params': variables,
            'expt_dir': self.output_dir,
            'expt_name': self.experiment_name,
            'resource': resource_name,
            'status': None,  # TODO: before: 'new'
            'submit time': time.time(),
            'start time': None,
            'end time': None,
            'driver_name': self.driver_name,
        }

        self.save_job(job)

        return job

    def remove_jobs(self):
        """Remove jobs from the jobs database."""
        self.db.remove(
            self.experiment_name,
            'jobs_' + self.driver_name,
            str(self.batch_number),
            {'expt_dir': self.output_dir, 'expt_name': self.experiment_name},
        )

    def all_jobs_finished(self):
        """Determine whether all jobs are finished.

        Finished can either mean, complete or failed

        Returns:
            bool: returns true if all jobs in the database have reached completion
                  or failed
        """
        num_pending = self.count_jobs({"status": "pending"})

        if (num_pending == self.num_pending) or (self.num_pending is None):
            pass
        else:
            self.num_pending = num_pending
            self.print_resources_status()

        if num_pending != 0:
            return False

        self.print_resources_status()
        return True

    def get_output_data(self, num_samples, gradient_bool):
        """Extract output data from database and return it.

        Args:
            num_samples (int): Number of evaluated samples
            gradient_bool (bool): Flag to determine whether the gradient
                                  of the model output w.r.t. to the input
                                  is expected (True if yes)

        Returns:
            dict: output dictionary; i
                  key:   | value:
                  'mean' | ndarray shape(batch_size, shape_of_response)
                  'var'  | ndarray (optional)
        """
        output = {}
        mean_values = []
        gradient_values = []
        if not self.all_jobs_finished():
            print("Not all jobs are finished yet, try again later")
        else:
            jobs = self.load_jobs(
                field_filters={'expt_dir': self.output_dir, 'expt_name': self.experiment_name}
            )

            # Sort job IDs in ascending order to match ordering of samples
            jobids = [job['id'] for job in jobs]
            jobids.sort()

            for ID in jobids:
                current_job = next(job for job in jobs if job['id'] == ID)
                mean_value = np.squeeze(current_job['result'])
                gradient_value = np.squeeze(current_job.get('gradient', None))

                if not mean_value.shape:
                    mean_value = np.expand_dims(mean_value, axis=0)
                    gradient_value = np.expand_dims(gradient_value, axis=0)

                mean_values.append(mean_value)
                gradient_values.append(gradient_value)

        output['mean'] = np.array(mean_values)[-num_samples:]
        if gradient_bool:
            output['gradient'] = np.array(gradient_values)[-num_samples:]

        return output

    # -------------private helper methods ---------------- #

    def _manage_restart(self, samples):
        """Manage different steps of restart.

        First, check if all results are in the database. Then, perform restart for missing results.
        Next, find and perform block-restart. And finally, load missing jobs into database and
        perform remaining restarts if necessary.

        Args:
            samples (DataFrame):     realization/samples of QUEENS simulation input variables

        Returns:
            jobid_for_data_processor(ndarray): jobids for data-processing
        """
        # Job that need direct data-processing
        jobid_for_data_processor = np.empty(shape=0)

        # Check results in database
        number_of_results_in_db, jobid_missing_results_in_db = self._check_results_in_db(samples)

        # All job results in database
        if number_of_results_in_db == samples.shape[0]:
            print(f"All results found in database.")

        # Not all job results in database
        else:
            # Run jobs with missing results in database
            if len(jobid_missing_results_in_db) > 0:
                self._manage_job_submission(samples, jobid_missing_results_in_db)
                jobid_for_data_processor = np.append(
                    jobid_for_data_processor, jobid_missing_results_in_db
                )

            # Find index for block-restart and run jobs
            jobid_for_block_restart = self._find_block_restart(samples)
            if jobid_for_block_restart is not None:
                range_block_restart = range(jobid_for_block_restart, samples.shape[0] + 1)
                self._manage_job_submission(samples, range_block_restart)
                jobid_for_data_processor = np.append(jobid_for_data_processor, range_block_restart)

            # Check if database is complete: all jobs are loaded
            is_every_job_in_db, jobid_smallest_in_db = self._check_jobs_in_db()

            # Load missing jobs into database and restart single failed jobs
            if not is_every_job_in_db:
                jobid_for_single_restart = self._load_missing_jobs_to_db(
                    samples, jobid_smallest_in_db
                )
                # Restart single failed jobs
                if len(jobid_for_single_restart) > 0:
                    self._manage_job_submission(samples, jobid_for_single_restart)
                    jobid_for_data_processor = np.append(
                        jobid_for_data_processor, jobid_for_single_restart
                    )

        return jobid_for_data_processor

    def _manage_jobs(self, samples):
        """Manage regular submission of jobs without restart.

        Args:
            samples (DataFrame): realization/samples of QUEENS simulation input variables

        Returns:
            jobid_for_data_processor(ndarray): jobids for data-processing
        """
        num_jobs = self.count_jobs()
        if not num_jobs or self.batch_number == 1:
            job_ids_generator = range(1, samples.shape[0] + 1, 1)
        else:
            job_ids_generator = range(num_jobs + 1, num_jobs + samples.shape[0] + 1, 1)

        self._manage_job_submission(samples, job_ids_generator)

        return np.array(job_ids_generator)

    def _check_results_in_db(self, samples):
        """Check complete results in database.

        Args:
            samples (DataFrame): realization/samples of QUEENS simulation input variables

        Returns:
            number_of_results_in_db (int):              number of results in database
            jobid_missing_results_in_db (ndarray):      job IDs of jobs with missing results
        """
        jobs = self.load_jobs(
            field_filters={'expt_dir': self.output_dir, 'expt_name': self.experiment_name}
        )

        number_of_results_in_db = 0
        jobid_missing_results_in_db = []
        for job in jobs:
            if job.get('result', np.empty(shape=0)).size != 0:
                number_of_results_in_db += 1
            else:
                jobid_missing_results_in_db = np.append(jobid_missing_results_in_db, job['id'])

        # Restart single failed jobs
        if len(jobid_missing_results_in_db) == 0:
            print('>> No single restart detected from database.')
        else:
            print(
                f'>> Single restart detected from database for job(s) #',
                jobid_missing_results_in_db.astype(int),
                '.',
                sep='',
            )
            jobid_missing_results_in_db = self._get_user_input_for_restart(
                samples, jobid_missing_results_in_db
            )

        return number_of_results_in_db, jobid_missing_results_in_db

    @staticmethod
    def _get_user_input_for_restart(samples, jobid_for_restart):
        """Ask the user to confirm the detected job ID(s) for restarts.

        Examples:
            Possible user inputs:
            y               confirm
            n               abort
            int             job ID
            int int int     several job IDs

        Args:
            samples (DataFrame):     realization/samples of QUEENS simulation input variables
            jobid_for_restart (int):    job ID(s) detected for restart

        Returns:
            jobid_for_restart:  ID(s) of the job(s) which the user wants to restart
        """
        print('>> Would you like to proceed?')
        print('>> Alternatively please type the ID of the job from which you want to restart!')
        print('>> Type "n" to abort.')

        while True:
            try:
                print('>> Please type "y", "n" or job ID(s) (int) >> ')
                answer = request_user_input_with_default_and_timeout(default="y", timeout=10)
            except SyntaxError:
                answer = None

            if answer.lower() == 'y':
                return jobid_for_restart
            elif answer.lower() == 'n':
                return []
            elif answer is None:
                print('>> Empty input! Only "y", "n" or job ID(s) (int) are valid inputs!')
                print('>> Try again!')
            else:
                try:
                    jobid_from_user = int(answer)
                    if jobid_from_user <= samples.shape[0]:
                        print(f'>> You chose a restart from job {jobid_from_user}.')
                        jobid_for_restart = jobid_from_user
                        return jobid_for_restart
                    else:
                        print(f'>> Your chosen job ID {jobid_from_user} is out of range.')
                        print('>> Try again!')
                except ValueError:
                    try:
                        jobid_from_user = np.array([int(jobid) for jobid in answer.split()])
                        valid_id = True
                        for jobid in jobid_from_user:
                            if jobid <= samples.shape[0]:
                                valid_id = True
                                pass
                            else:
                                valid_id = False
                                print(f'>> Your chosen job ID {jobid} is out of range.')
                                print('>> Try again!')
                                break
                        if valid_id:
                            print(f'>> You chose a restart of jobs {jobid_from_user}.')
                            return jobid_from_user
                    except IndexError:
                        print(
                            f'>> The input "{answer}" is not an appropriate choice! '
                            f'>> Only "y", "n" or a job ID(s) (int) are valid inputs!'
                        )
                        print('>> Try again!')

    def _check_jobs_in_db(self):
        """Check jobs in database.

        Find the job with the smallest job ID in the database.

        Returns:
            is_every_job_in_db (boolean):   true if smallest job ID in database is 1
            jobid_smallest_in_db (int):     smallest job ID in database
        """
        jobs = self.load_jobs(
            field_filters={'expt_dir': self.output_dir, 'expt_name': self.experiment_name}
        )

        jobid_smallest_in_db = min([job['id'] for job in jobs])
        is_every_job_in_db = jobid_smallest_in_db == 1

        return is_every_job_in_db, jobid_smallest_in_db

    def _check_job_completions(self, jobid_range):
        """Check AWS tasks to determine completed jobs."""
        jobs = self.load_jobs(
            field_filters={'expt_dir': self.output_dir, 'expt_name': self.experiment_name}
        )
        for check_jobid in jobid_range:
            for _, resource in self.resources.items():
                try:
                    current_check_job = next(job for job in jobs if job['id'] == check_jobid)
                    if current_check_job['status'] != 'complete':
                        completed, failed = resource.check_job_completion(current_check_job)

                        # determine if this a failed job and return if yes
                        if failed:
                            current_check_job['status'] = 'failed'
                            return

                        # determine if this a completed job and return if yes
                        if completed:
                            current_check_job['status'] = 'complete'
                            current_check_job['end time'] = time.time()
                            computing_time = (
                                current_check_job['end time'] - current_check_job['start time']
                            )
                            _logger.info(
                                'Successfully completed job {:d} (No. of proc.: {:d}, '
                                'computing time: {:08.2f} s).\n'.format(
                                    current_check_job['id'],
                                    current_check_job['num_procs'],
                                    computing_time,
                                )
                            )
                            self.save_job(current_check_job)
                            return

                except (StopIteration, IndexError):
                    pass

    def _find_block_restart(self, samples):
        """Find index for block-restart.

        Args:
            samples (DataFrame):     realization/samples of QUEENS simulation input variables

        Returns:
            jobid_for_block_restart (int):  index for block-restart of failed jobs
        """
        # Find number of subdirectories in output directory
        if self.remote_connect:
            command_list = [
                'ssh',
                self.remote_connect,
                '"cd',
                self.output_dir,
                '; ls -l | grep ' '"^d" | wc -l "',
            ]
        else:
            command_list = ['cd', self.output_dir, '; ls -l | grep ' '"^d" | wc -l']
        command_string = ' '.join(command_list)
        _, _, str_number_of_subdirectories, _ = run_subprocess(command_string)
        number_of_subdirectories = (
            int(str_number_of_subdirectories) if str_number_of_subdirectories else 0
        )
        if not number_of_subdirectories:
            raise FileNotFoundError(
                "You chose restart_from_finished simulations, but your output folder is empty."
            )

        if number_of_subdirectories < samples.shape[0]:
            # Start from (number of subdirectories) + 1
            jobid_start_search = int(number_of_subdirectories) + 1
        else:
            jobid_start_search = samples.shape[0]

        jobid_for_block_restart = None
        jobid_for_restart_found = False
        # Loop backwards to find first completed job
        for jobid in range(jobid_start_search, 0, -1):
            # Loop over all available resources
            for resource_name, resource in self.resources.items():
                current_job = self._get_current_restart_job(jobid, resource, resource_name, samples)

                if current_job.get('result', np.empty(shape=0)).size != 0:
                    # Last finished job -> restart from next job
                    jobid_for_block_restart = jobid + 1
                    jobid_for_restart_found = True
                    break

            if jobid_for_restart_found:
                break
            elif jobid == 1:
                raise RuntimeError(
                    'Block-restart not found. Check for errors in DataProcessor-Module'
                )

        # If jobid for block-restart out of range -> no restart
        if jobid_for_block_restart > samples.shape[0]:
            jobid_for_block_restart = None

        # Get user input for block-restart
        if jobid_for_block_restart is not None:
            print(f'>> Block-restart detected for job #{jobid_for_block_restart}. ')
            jobid_for_block_restart = self._get_user_input_for_restart(
                samples, jobid_for_block_restart
            )
            if (not isinstance(jobid_for_block_restart, int)) and (
                jobid_for_block_restart is not None
            ):
                raise AssertionError('Only one job ID allowed for block-restart. ')
        else:
            print('>> No block-restart detected.')

        return jobid_for_block_restart

    def _load_missing_jobs_to_db(self, samples, jobid_end):
        """Load missing jobs to database 1, ..., jobid_end.

        Args:
            samples (DataFrame):     realization/samples of QUEENS simulation input variables
            jobid_end (int):         index of job where to stop loading results

        Returns:
            jobid_for_single_restart:     array with indices of failed jobs and missing results
        """
        jobid_for_single_restart = []

        for jobid in range(1, jobid_end):
            for resource_name, resource in self.resources.items():
                current_job = self._get_current_restart_job(jobid, resource, resource_name, samples)

                if current_job.get('result', np.empty(shape=0)).size == 0:
                    # No result
                    jobid_for_single_restart = np.append(jobid_for_single_restart, int(jobid))

        # Get user input for restart of single jobs
        if len(jobid_for_single_restart) == 0:
            print('>> No restart of single jobs detected.')
            self.print_resources_status()
        else:
            print(
                f'>> Single restart detected for job(s) #',
                jobid_for_single_restart.astype(int),
                '.',
                sep='',
            )
            jobid_for_single_restart = self._get_user_input_for_restart(
                samples, jobid_for_single_restart
            )

        return jobid_for_single_restart

    def _manage_job_submission(self, samples, jobid_range):
        """Iterate over samples and manage submission of jobs.

        Args:
            samples (DataFrame):     realization/samples of QUEENS simulation input variables
            jobid_range (range):     range of job IDs which are submitted
        """
        for jobid in jobid_range:
            processed_suggestion = False
            while not processed_suggestion:
                # Loop over all available resources
                for resource_name, resource in self.resources.items():
                    num_pending_jobs_of_resource = self.count_jobs(
                        {"status": "pending", "resource": resource_name}
                    )
                    if resource.accepting_jobs(num_pending_jobs=num_pending_jobs_of_resource):
                        # wait with submission of next job
                        if self.time_for_data_copy is not None:
                            time.sleep(self.time_for_data_copy)
                        # try to load existing job (with same jobid) from the database
                        current_job = self.load_jobs(
                            field_filters={
                                'id': jobid,
                                'expt_dir': self.output_dir,
                                'expt_name': self.experiment_name,
                            }
                        )
                        if len(current_job) == 1:
                            current_job = current_job[0]
                        elif not current_job:
                            if self._internal_batch_state != self.batch_number:
                                self._internal_batch_state = self.batch_number
                                counter = 0

                            counter += 1
                            job_num = counter
                            sample_dict = self.parameters.sample_as_dict(samples[job_num - 1])
                            current_job = self.create_new_job(sample_dict, resource_name, jobid)
                        else:
                            raise ValueError(f"Found more than one job with jobid {jobid} in db.")

                        current_job['status'] = 'pending'
                        self.save_job(current_job)

                        # Submit the job to the appropriate resource
                        # this is the actual job submission
                        this.restart_flag = False
                        process_id = self.attempt_dispatch(resource, current_job)

                        # Set the status of the job appropriately (successfully submitted or not)
                        if process_id is None:
                            current_job['status'] = 'broken'
                        else:
                            current_job['status'] = 'pending'
                            current_job['proc_id'] = process_id

                        processed_suggestion = True
                        self.print_resources_status()

                    else:
                        time.sleep(self.polling_time)
                        # check job completions for jobscript-based native driver
                        for _, resource in self.resources.items():
                            if self.direct_scheduling:
                                self._check_job_completions(jobid_range)

        return

    def _get_current_restart_job(self, jobid, resource, resource_name, samples):
        """Get the current job from database.

        Get job with ID (job_id) from database or from output directory.

        Args:
            jobid (int):      job ID
            resource (Resource object): computing resource
            resource_name (str):  name of computing resource
            samples (DataFrame):     realization/samples of QUEENS simulation input variables

        Returns:
            current job (dict):    current job with ID (job_id)
        """
        # try to load existing job (with same jobid) from the database
        current_job = self.load_jobs(
            field_filters={
                'id': jobid,
                'expt_dir': self.output_dir,
                'expt_name': self.experiment_name,
            }
        )

        if not current_job:
            # job not in database -> load result from output folder
            if self._internal_batch_state != self.batch_number:
                self._internal_batch_state = self.batch_number
                counter = 0

            counter += 1
            job_num = counter

            sample_dict = self.parameters.sample_as_dict(samples[job_num - 1])
            current_job = self.create_new_job(sample_dict, resource_name, jobid)

            this.restart_flag = True
            self.attempt_dispatch(resource, current_job)
            current_job = self.load_jobs(
                field_filters={
                    'id': jobid,
                    'expt_dir': self.output_dir,
                    'expt_name': self.experiment_name,
                }
            )
        elif len(current_job) != 1:
            raise ValueError(f"Found more than one job with jobid {jobid} in db.")
        return current_job[0]

    def _manage_data_processor_submission(self, jobid_range):
        """Manage submission of data processing.

        Args:
            jobid_range (range):     range of job IDs which are submitted
        """
        jobs = self.load_jobs(
            field_filters={'expt_dir': self.output_dir, 'expt_name': self.experiment_name}
        )
        for jobid in jobid_range:
            for _, resource in self.resources.items():
                try:
                    current_job = next(job for job in jobs if job['id'] == jobid)
                except (StopIteration, IndexError):
                    pass

                resource.dispatch_data_processor_job(self.batch_number, current_job)

        self.print_resources_status()

        return

    def print_resources_status(self):
        """Print out whats going on on the resources."""
        _logger.info('\nResources:      ')
        left_indent = 16
        indentation = ' ' * left_indent
        _logger.info('NAME            PENDING      COMPLETED    FAILED   \n')
        _logger.info(indentation)
        _logger.info('------------    ---------    ---------    ---------\n')
        total_pending = 0
        total_complete = 0
        total_failed = 0

        # for resource in resources:
        for resource_name, resource in self.resources.items():
            pending = self.count_jobs({"status": "pending", "resource": resource_name})
            complete = self.count_jobs({"status": "complete", "resource": resource_name})
            failed = self.count_jobs({"status": "failed", "resource": resource_name})
            total_pending += pending
            total_complete += complete
            total_failed += failed
            _logger.info(
                '{}{:12.12}    {:<9d}    {:<9d}    {:<9d}\n'.format(
                    indentation, resource.name, pending, complete, failed
                )
            )
        _logger.info(
            '{}{:12.12}    {:<9d}    {:<9d}    {:<9d}\n'.format(
                indentation, '*TOTAL*', total_pending, total_complete, total_failed
            )
        )
        _logger.info('\n')
