"""Job interface class."""
import logging
import pathlib
import time

import numpy as np

import pqueens.database.database as DB_module
from pqueens.interfaces.interface import Interface
from pqueens.resources.resource import parse_resources_from_configuration

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
        output_dir (path):                       directory to write output to
        parameters (dict):                       dictionary with parameters
        time_for_data_copy (float): Time (s) to wait such that copying process of simulation
                                    input file can finish, and we do not overload the network
        job_num (int):              Number of the current job
        _internal_batch_state (int): Helper attribute to compare batch_number with the internal
                                     batch state to detect changes in the batch number.
    """

    def __init__(
        self,
        interface_name,
        resources,
        experiment_name,
        db,
        polling_time,
        output_dir,
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
            output_dir (path):          directory to write output to
            remote (bool):              true of remote computation
            remote_connect (str):       connection to computing resource
            scheduler_type (str):       scheduler type
            direct_scheduling (bool):   true if direct scheduling
            time_for_data_copy (float): Time (s) to wait such that copying process of simulation
                                        input file can finish, and we do not overload the network
            driver_name (str):          Name of the associated driver for the current interface
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
        self.remote = remote
        self.remote_connect = remote_connect
        self.scheduler_type = scheduler_type
        self.direct_scheduling = direct_scheduling
        self.time_for_data_copy = time_for_data_copy
        self.driver_name = driver_name
        self._internal_batch_state = 0
        self.job_num = 0

    @classmethod
    def from_config_create_interface(cls, interface_name, config):
        """Create JobInterface from config dictionary.

        Args:
            interface_name (str):   name of interface
            config (dict):          dictionary containing problem description

        Returns:
            interface:              instance of JobInterface
        """
        # get experiment name and polling time
        experiment_name = config['global_settings']['experiment_name']
        polling_time = config.get('polling-time', 1.0)

        interface_options = config[interface_name]
        driver_name = interface_options.get('driver', None)
        if driver_name is None:
            raise Exception("No driver_name specified for the JobInterface.")

        # get resources from config
        resources = parse_resources_from_configuration(config, driver_name)

        # get various scheduler options
        # TODO: This is not nice
        first = list(config['resources'])[0]
        scheduler_name = config['resources'][first]['scheduler']
        scheduler_options = config[scheduler_name]
        output_dir = pathlib.Path(scheduler_options["experiment_dir"])
        scheduler_type = scheduler_options['scheduler_type']

        # get flag for remote scheduling
        if scheduler_options.get('remote'):
            remote = True
            remote_connect = scheduler_options['remote']['connect']
        else:
            remote = False
            remote_connect = None

        # get flag for Singularity
        singularity = scheduler_options.get('singularity', False)
        if not isinstance(singularity, bool):
            raise TypeError("Singularity option has to be a boolean (true or false).")

        # set flag for direct scheduling
        direct_scheduling = False
        if not singularity:
            if (
                scheduler_type == 'pbs'
                or scheduler_type == 'slurm'
                or (scheduler_type == 'standard' and remote)
            ):
                direct_scheduling = True

        db = DB_module.database

        # get waiting time for copying data
        time_for_data_copy = interface_options.get('time_for_data_copy')

        # instantiate object
        return cls(
            interface_name,
            resources,
            experiment_name,
            db,
            polling_time,
            output_dir,
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
        jobid_for_data_processor = self._manage_jobs(samples)

        # Post run
        for _ in self.resources:
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

    def attempt_dispatch(self, resource, new_job):
        """Attempt to dispatch job multiple times.

        This is the actual job submission.
        Submitting jobs to the queue sometimes fails, hence we try multiple times
        before giving up. We also wait half a second between submit commands

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

    def count_jobs(self, field_filters=None):
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

    def load_jobs(self, field_filters=None):
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
            {'id': job['id'], 'expt_dir': str(self.output_dir), 'expt_name': self.experiment_name},
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
            'expt_dir': str(self.output_dir),
            'expt_name': self.experiment_name,
            'resource': resource_name,
            'status': "",  # TODO: before: 'new'
            'submit_time': time.time(),
            'start_time': 0.0,
            'end_time': 0.0,
            'driver_name': self.driver_name,
        }

        self.save_job(job)

        return job

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
                field_filters={'expt_dir': str(self.output_dir), 'expt_name': self.experiment_name}
            )

            # Sort job IDs in ascending order to match ordering of samples
            jobids = [job['id'] for job in jobs]
            jobids.sort()

            for current_job_id in jobids:
                current_job = next(job for job in jobs if job['id'] == current_job_id)
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

    def _manage_jobs(self, samples):
        """Manage regular submission of jobs.

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

    def _check_job_completions(self, jobid_range):
        """Check AWS tasks to determine completed jobs."""
        jobs = self.load_jobs(
            field_filters={'expt_dir': str(self.output_dir), 'expt_name': self.experiment_name}
        )
        for check_jobid in jobid_range:
            for resource in self.resources.values():
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
                            current_check_job['end_time'] = time.time()
                            computing_time = (
                                current_check_job['end_time'] - current_check_job['start_time']
                            )
                            _logger.info(
                                f'Successfully completed job {current_check_job["id"]} '
                                f'(No. of proc.: {current_check_job["num_procs"]}, '
                                f'computing time: {computing_time} s).\n'
                            )
                            self.save_job(current_check_job)
                            return

                except (StopIteration, IndexError):
                    pass

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
                                'expt_dir': str(self.output_dir),
                                'expt_name': self.experiment_name,
                            }
                        )
                        if len(current_job) == 1:
                            current_job = current_job[0]
                        elif not current_job:
                            if self._internal_batch_state != self.batch_number:
                                self._internal_batch_state = self.batch_number
                                self.job_num = 0

                            self.job_num += 1
                            sample_dict = self.parameters.sample_as_dict(samples[self.job_num - 1])
                            current_job = self.create_new_job(sample_dict, resource_name, jobid)
                        else:
                            raise ValueError(f"Found more than one job with jobid {jobid} in db.")

                        current_job['status'] = 'pending'
                        self.save_job(current_job)

                        # Submit the job to the appropriate resource
                        # this is the actual job submission
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
                        for _ in self.resources:
                            if self.direct_scheduling:
                                self._check_job_completions(jobid_range)

    def _manage_data_processor_submission(self, jobid_range):
        """Manage submission of data processing.

        Args:
            jobid_range (range):     range of job IDs which are submitted
        """
        jobs = self.load_jobs(
            field_filters={'expt_dir': str(self.output_dir), 'expt_name': self.experiment_name}
        )
        for jobid in jobid_range:
            for resource in self.resources.values():
                try:
                    current_job = next(job for job in jobs if job['id'] == jobid)
                except (StopIteration, IndexError):
                    pass

                resource.dispatch_data_processor_job(self.batch_number, current_job)

        self.print_resources_status()

    def print_resources_status(self):
        """Print out whats going on on the resources."""
        _logger.info('\nResources:      ')
        left_indent = 16
        indentation = ' ' * left_indent
        _logger.info('NAME            PENDING      COMPLETED    FAILED   ')
        _logger.info('------------    ---------    ---------    ---------')
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
                f'{resource.name:12.12}    {pending:<9d}    {complete:<9d}    {failed:<9d}'
            )
        _logger.info(
            f'{"*TOTAL*":12.12}    {total_pending:<9d}    {total_complete:<9d}    '
            f'{total_failed:<9d}'
        )
        _logger.info('\n')
