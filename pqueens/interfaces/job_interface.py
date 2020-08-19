import time
import numpy as np
import pandas as pd
import os
import sys
from pqueens.interfaces.interface import Interface
from pqueens.resources.resource import parse_resources_from_configuration
from pqueens.resources.resource import print_resources_status
from pqueens.database.mongodb import MongoDB
from pqueens.utils.run_subprocess import run_subprocess
from pqueens.utils.aws_output_string_extractor import aws_extract
from pqueens.utils.user_input import request_user_input_with_default_and_timeout

this = sys.modules[__name__]
this.restart_flag = None


class JobInterface(Interface):
    """
        Class for mapping input variables to responses

        The JobInterface class maps input variables to outputs, i.e. responses
        by creating a job which is then submitted to a job manager on some
        local or remote resource, which in turn then actually runs the
        simulation software.

    Attributes:
        interface_name (string):                 name of interface
        resources (dict):                        dictionary with resources
        experiment_name (string):                name of experiment
        db_address (string):                     address of database to use
        db (mongodb):                            mongodb to store results and job info
        polling_time (int):                      how frequently do we check if jobs are done
        output_dir (string):                     directory to write output to
        restart_from_finished_simulation (bool): true if restart option is chosen
        parameters (dict):                       dictionary with parameters
        connect (string):                        connection to computing resource

    """

    def __init__(
        self,
        interface_name,
        resources,
        experiment_name,
        db,
        polling_time,
        output_dir,
        restart_from_finished_simulation,
        parameters,
        connect,
        scheduler_type,
        direct_scheduling,
    ):
        """ Create JobInterface

        Args:
            interface_name (string):    name of interface
            resources (dict):           dictionary with resources
            experiment_name (string):   name of experiment
            db (mongodb):               mongodb to store results and job info
            polling_time (int):         how frequently do we check if jobs are done
            output_dir (string):        directory to write output to
            parameters (dict):          dictionary with parameters
            restart_from_finished_simulation (bool): true if restart option is chosen
            connect (string):  connection to computing resource
        """
        self.name = interface_name
        self.resources = resources
        self.experiment_name = experiment_name
        self.db = db
        self.polling_time = polling_time
        self.output_dir = output_dir
        self.parameters = parameters
        self.batch_number = 0
        self.num_pending = None
        self.restart_from_finished_simulation = restart_from_finished_simulation
        self.connect_to_resource = connect
        self.scheduler_type = scheduler_type
        self.direct_scheduling = direct_scheduling

    @classmethod
    def from_config_create_interface(cls, interface_name, config):
        """ Create JobInterface from config dictionary

        Args:
            interface_name (str):   name of interface
            config (dict):          dictionary containing problem description

        Returns:
            interface:              instance of JobInterface
        """
        # get resources from config
        resources = parse_resources_from_configuration(config)

        # get experiment name
        experiment_name = config['global_settings']['experiment_name']

        restart_from_finished_simulation = config['driver']['driver_params'].get(
            'restart_from_finished_simulation', False
        )

        # establish new database for this QUEENS run and
        # potentially drop other databases
        db = MongoDB.from_config_create_database(config)

        # print out database information
        db.print_database_information(restart=restart_from_finished_simulation)

        polling_time = config.get('polling-time', 1.0)

        # TODO: This is not nice -> should be solved solemnly over Driver class
        output_dir = config['driver']['driver_params']["experiment_dir"]

        # TODO: This is not nice
        connect = list(resources.values())[0].scheduler.connect_to_resource

        # TODO: This is definitely not nice, either
        first = list(config['resources'])[0]
        scheduler_name = config['resources'][first]['scheduler']
        scheduler_type = config[scheduler_name]['scheduler_type']

        if (
            scheduler_type == 'ecs_task'
            or scheduler_type == 'local_pbs'
            or scheduler_type == 'local_slurm'
        ):
            direct_scheduling = True
        else:
            direct_scheduling = False

        parameters = config['parameters']

        # instantiate object
        return cls(
            interface_name,
            resources,
            experiment_name,
            db,
            polling_time,
            output_dir,
            restart_from_finished_simulation,
            parameters,
            connect,
            scheduler_type,
            direct_scheduling,
        )

    def map(self, samples):
        """
        Mapping function which orchestrates call to external simulation software

        Second variant which takes the input samples as argument

        Args:
            samples (list):         realization/samples of QUEENS simulation input variables

        Returns:
            np.array,np.array       two arrays containing the inputs from the
                                    suggester, as well as the corresponding outputs

        """
        self.batch_number += 1

        # Convert samples to pandas DataFrame to use index
        samples = pd.DataFrame(samples, index=range(1, len(samples) + 1))

        job_manager = self.get_job_manager()
        jobid_for_post_post = job_manager(samples)

        # Post run
        for _, resource in self.resources.items():
            # only for ECS task scheduler and jobscript-based native driver
            if self.direct_scheduling and jobid_for_post_post.size != 0:
                # check tasks to determine completed jobs
                while not self.all_jobs_finished():
                    time.sleep(self.polling_time)
                    self._check_job_completions(jobid_for_post_post)

                # submit post-post jobs
                self._manage_post_post_submission(jobid_for_post_post)

            # for all other resources:
            else:
                # just wait for all jobs to finish
                while not self.all_jobs_finished():
                    time.sleep(self.polling_time)

                # only for remote computation:
                resource.scheduler.post_run()

        # get sample and response data
        return self.get_output_data()

    def get_job_manager(self):
        """Decide whether or not restart is performed.

        Returns:
            function object:    management function which should be used

        """
        if self.restart_from_finished_simulation:
            return self._manage_restart

        else:
            return self._manage_jobs

    def attempt_dispatch(self, resource, new_job):
        """ Attempt to dispatch job multiple times

        Submitting jobs to the queue sometimes fails, hence we try multiple times
        before giving up. We also wait one second between submit commands

        Args:
            resource (resource object): Resource to submit job to
            new_job (dict):             Dictionary with job

        Returns:
            int: Process ID of submitted job if successfull, None otherwise
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

    def load_jobs(self):
        """ Load jobs from the jobs database

        Returns:
            list : list with all jobs or an empty list
        """

        jobs = []
        for batch_num in range(1, self.batch_number + 1):
            job = self.db.load(
                self.experiment_name,
                str(batch_num),
                'jobs',
                {'expt_dir': self.output_dir, 'expt_name': self.experiment_name},
            )
            if isinstance(job, list):
                jobs.extend(job)
            else:
                if job is not None:
                    jobs.append(job)

        return jobs

    def save_job(self, job):
        """ Save a job to the job database

        Args:
            job (dict): dictionary with job details
        """
        self.db.save(
            job,
            self.experiment_name,
            'jobs',
            str(self.batch_number),
            {'id': job['id'], 'expt_dir': self.output_dir, 'expt_name': self.experiment_name},
        )

    def create_new_job(self, variables, resource_name, new_id=None):
        """ Create new job and save it to database and return it

        Args:
            variables (Variables):     variables to run model at
            resource_name (string):     name of resource
            new_id (int):                  id for job

        Returns:
            job: new job
        """

        jobs = self.load_jobs()
        if new_id is None:
            print("Created new job")
            job_id = len(jobs) + 1
        else:
            job_id = int(new_id)

        job = {
            'id': job_id,
            'params': variables.get_active_variables(),
            'expt_dir': self.output_dir,
            'expt_name': self.experiment_name,
            'resource': resource_name,
            'status': None,  # TODO: before: 'new'
            'submit time': time.time(),
            'start time': None,
            'end time': None,
        }

        self.save_job(job)

        return job

    def remove_jobs(self):
        """ Remove jobs from the jobs database

        """
        self.db.remove(
            self.experiment_name,
            'jobs',
            str(self.batch_number),
            {'expt_dir': self.output_dir, 'expt_name': self.experiment_name},
        )

    def tired(self, resource):
        """ Quick check whether a resource is fully occupied

        Args:
            (resource): resource object
        Returns:
            bool: whether or not resource is tired
        """
        jobs = self.load_jobs()
        if resource.accepting_jobs(jobs):
            return False
        return True

    def all_jobs_finished(self):
        """ Determine whether all jobs are finished

        Finished can either mean, complete or failed

        Returns:
            bool: returns true if all jobs in the database have reached completion
                  or failed
        """
        jobs = self.load_jobs()
        num_pending = 0
        for job in jobs:
            if (
                job['status'] != 'complete'
                and job['status'] != 'failed'
                and job['status'] != 'broken'
            ):
                num_pending = num_pending + 1

        if (num_pending == self.num_pending) or (self.num_pending is None):
            pass
        else:
            self.num_pending = num_pending
            print_resources_status(self.resources, jobs)

        if num_pending != 0:
            return False

        print_resources_status(self.resources, jobs)
        return True

    def get_output_data(self):
        """ Extract output data from database and return it

        Returns:
            dict: output dictionary; i
                  key:   | value:
                  'mean' | ndarray shape(batch_size, shape_of_response)
                  'var'  | ndarray (optional)

        """
        output = {}
        mean_values = []
        if not self.all_jobs_finished():
            print("Not all jobs are finished yet, try again later")
        else:
            jobs = self.load_jobs()

            # Sort job IDs in ascending order to match ordering of samples
            jobids = [job['id'] for job in jobs]
            jobids.sort()

            for ID in jobids:
                current_job = next(job for job in jobs if job['id'] == ID)
                mean_value = np.squeeze(current_job['result'])
                if not mean_value.shape:
                    mean_value = np.expand_dims(mean_value, axis=0)
                mean_values.append(mean_value)

        output['mean'] = np.array(mean_values)

        return output

    # -------------private helper methods ---------------- #

    def _manage_restart(self, samples):
        """Manage different steps of restart.

        First, check if all results are in the database. Then, perform restart for missing results.
        Next, find and perform block-restart. And finally, load missing jobs into database and
        perform remaining restarts if necessary.

        Args:
            samples (DataFrame):     realization/samples of QUEENS simulation input variables
        """
        # Job that need direct post-post-processing
        jobid_for_post_post = np.empty(shape=0)

        # Check results in database
        number_of_results_in_db, jobid_missing_results_in_db = self._check_results_in_db(samples)

        # All job results in database
        if number_of_results_in_db == samples.size:
            print(f"All results found in database.")

        # Not all job results in database
        else:
            # Run jobs with missing results in database
            if len(jobid_missing_results_in_db) > 0:
                self._manage_job_submission(samples, jobid_missing_results_in_db)
                jobid_for_post_post = np.append(jobid_for_post_post, jobid_missing_results_in_db)

            # Find index for block-restart and run jobs
            jobid_for_block_restart = self._find_block_restart(samples)
            if jobid_for_block_restart is not None:
                range_block_restart = range(jobid_for_block_restart, samples.size + 1)
                self._manage_job_submission(samples, range_block_restart)
                jobid_for_post_post = np.append(jobid_for_post_post, range_block_restart)

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
                    jobid_for_post_post = np.append(jobid_for_post_post, jobid_for_single_restart)

        return jobid_for_post_post

    def _manage_jobs(self, samples):
        """
        Manage regular submission of jobs without restart.

        Args:
            samples (DataFrame): realization/samples of QUEENS simulation input variables

        """
        jobs = self.load_jobs()
        if not jobs or self.batch_number == 1:
            job_ids_generator = range(1, samples.size + 1, 1)
        else:
            job_ids_generator = range(len(jobs) + 1, len(jobs) + samples.size + 1, 1)

        self._manage_job_submission(samples, job_ids_generator)

    def _check_results_in_db(self, samples):
        """Check complete results in database.

        Args:
            samples (DataFrame): realization/samples of QUEENS simulation input variables

        Returns:
            number_of_results_in_db (int):              number of results in database
            jobid_missing_results_in_db (ndarray):      job IDs of jobs with missing results
        """
        jobs = self.load_jobs()

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
                return None
            elif answer is None:
                print('>> Empty input! Only "y", "n" or job ID(s) (int) are valid inputs!')
                print('>> Try again!')
            else:
                try:
                    jobid_from_user = int(answer)
                    if jobid_from_user <= samples.size:
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
                            if jobid <= samples.size:
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
        """Check jobs in database and find the job with the smallest job ID in the database.

        Returns:
            is_every_job_in_db (boolean):   true if smallest job ID in database is 1
            jobid_smallest_in_db (int):     smallest job ID in database
        """
        jobs = self.load_jobs()

        jobid_smallest_in_db = min([job['id'] for job in jobs])
        is_every_job_in_db = jobid_smallest_in_db == 1

        return is_every_job_in_db, jobid_smallest_in_db

    def _check_job_completions(self, jobid_range):
        """Check AWS tasks to determine completed jobs.
        """
        jobs = self.load_jobs()
        for check_jobid in jobid_range:
            try:
                current_check_job = next(job for job in jobs if job['id'] == check_jobid)
                if current_check_job['status'] != 'complete':
                    completed = False
                    if self.scheduler_type == 'ecs_task':
                        command_list = [
                            "aws ecs describe-tasks ",
                            "--cluster worker-queens-cluster ",
                            "--tasks ",
                            current_check_job['aws_arn'],
                        ]
                        cmd = ''.join(filter(None, command_list))
                        _, _, stdout, stderr = run_subprocess(cmd)
                        if stderr:
                            current_check_job['status'] = 'failed'
                        status_str = aws_extract("lastStatus", stdout)
                        if status_str == 'STOPPED':
                            completed = True
                    else:
                        # indicate completion by existing control file in output directory
                        completed = os.path.isfile(current_check_job['control_file_path'])
                    if completed:
                        current_check_job['status'] = 'complete'
                        current_check_job['end time'] = time.time()
                        computing_time = (
                            current_check_job['end time'] - current_check_job['start time']
                        )
                        sys.stdout.write(
                            'Successfully completed job {:d} (No. of proc.: {:d}, '
                            'computing time: {:08.2f} s).\n'.format(
                                current_check_job['id'],
                                current_check_job['num_procs'],
                                computing_time,
                            )
                        )
                        self.save_job(current_check_job)
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
        if self.connect_to_resource:
            command_list = [
                'ssh',
                self.connect_to_resource,
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
        assert (
            number_of_subdirectories != 0
        ), "You chose restart_from_finished simulations, but your output folder is empty. "

        if number_of_subdirectories < samples.size:
            # Start from (number of subdirectories) + 1
            jobid_start_search = int(number_of_subdirectories) + 1
        else:
            jobid_start_search = samples.size

        jobid_for_block_restart = None
        jobid_for_restart_found = False
        # Loop backwards to find first completed job
        for jobid in range(jobid_start_search, -1, -1):
            # Loop over all available resources
            for resource_name, resource in self.resources.items():
                current_job, process_id = self._get_current_job(
                    samples, resource, resource_name, jobid
                )

                assert process_id == 0, "Error: Process ID in find_block_restart must be 0."

                if current_job.get('result', np.empty(shape=0)).size != 0:
                    # Last finished job -> restart from next job
                    jobid_for_block_restart = jobid + 1
                    jobid_for_restart_found = True
                    break

            if jobid_for_restart_found:
                break

        # If jobid for block-restart out of range -> no restart
        if jobid_for_block_restart > samples.size:
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
                current_job, process_id = self._get_current_job(
                    samples, resource, resource_name, jobid
                )

                assert process_id == 0, "Error: Process ID in load_missing_jobs_to_db must be 0."

                if current_job.get('result', np.empty(shape=0)).size == 0:
                    # No result
                    jobid_for_single_restart = np.append(jobid_for_single_restart, int(jobid))

        # Get user input for restart of single jobs
        if len(jobid_for_single_restart) == 0:
            print('>> No restart of single jobs detected.')
            jobs = self.load_jobs()
            print_resources_status(self.resources, jobs)
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
        jobs = self.load_jobs()

        for jobid in jobid_range:
            processed_suggestion = False
            while not processed_suggestion:
                # Loop over all available resources
                for resource_name, resource in self.resources.items():
                    if resource.accepting_jobs(jobs):
                        job_num = jobid - (self.batch_number - 1) * samples.size
                        variables = samples.loc[job_num][0]
                        try:
                            current_job = next(
                                job for job in jobs if job is not None and job['id'] == jobid
                            )
                        except (StopIteration, IndexError, KeyError):
                            current_job = self.create_new_job(variables, resource_name, jobid)

                        current_job['status'] = 'pending'
                        self.save_job(current_job)

                        # Submit the job to the appropriate resource
                        this.restart_flag = False
                        process_id = self.attempt_dispatch(resource, current_job)

                        # Set the status of the job appropriately (successfully submitted or not)
                        if process_id is None:
                            current_job['status'] = 'broken'
                        else:
                            current_job['status'] = 'pending'
                            current_job['proc_id'] = process_id

                        processed_suggestion = True
                        jobs = self.load_jobs()
                        print_resources_status(self.resources, jobs)

                    else:
                        time.sleep(self.polling_time)
                        # check job completions for ECS task scheduler and
                        # jobscript-based native driver
                        for _, resource in self.resources.items():
                            if self.direct_scheduling:
                                self._check_job_completions(jobid_range)
                        jobs = self.load_jobs()

        return

    def _manage_post_post_submission(self, jobid_range):
        """Manage submission of post-post processing.

        Args:
            jobid_range (range):     range of job IDs which are submitted
        """
        jobs = self.load_jobs()
        for jobid in jobid_range:
            for _, resource in self.resources.items():
                try:
                    current_job = next(job for job in jobs if job['id'] == jobid)
                except (StopIteration, IndexError):
                    pass

                resource.dispatch_post_post_job(self.batch_number, current_job)

        jobs = self.load_jobs()
        print_resources_status(self.resources, jobs)

        return

    def _get_current_job(self, samples, resource, resource_name, job_id):
        """Get the current job with ID (job_id) from database or from output directory.

        Args:
            samples (DataFrame):     realization/samples of QUEENS simulation input variables
            resource (Resource object): computing resource
            resource_name (str):  name of computing resource
            job_id (int):      job ID

        Returns:
            current job (dict):    current job with ID (job_id)
            process_id (int):      0 if loaded from database or output directory
        """
        variables = samples.loc[job_id][0]
        jobs = self.load_jobs()

        try:
            # Job already in database
            current_job = next(job for job in jobs if job['id'] == job_id)
            process_id = 0
        except StopIteration:
            # Job not in database:
            # load result from output directory into database
            current_job = self.create_new_job(variables, resource_name, job_id)
            this.restart_flag = True
            process_id = self.attempt_dispatch(resource, current_job)

            jobs = self.load_jobs()
            current_job = next(job for job in jobs if job['id'] == job_id)

        return current_job, process_id
