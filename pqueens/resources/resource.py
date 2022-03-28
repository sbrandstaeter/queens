"""QUEENS resource module.

This module contains everything that is necessary to manage computing
resources.A computing ressource can be a single machine or a HPC
cluster. The resource can provide basic status information as well as
workload capacity. If the workload capacity allows it the computing
resource accepts jobs and executes them.
"""
import sys

import numpy as np

from pqueens.schedulers import from_config_create_scheduler

# TODO refactor this method into a class method


def parse_resources_from_configuration(config, driver_name):
    """Parse the configuration dictionary.

    Args:
        config (dict): Dictionary with problem description
        driver_name (str): Name of driver that should be used in this job-submission

    Returns:
        dict: Dictionary with resource objects keyed with resource name
    """
    if "resources" in config:
        resources = dict()
        for resource_name, _ in config["resources"].items():
            global_settings = config.get("global_settings")
            exp_name = global_settings.get("experiment_name")
            resources[resource_name] = resource_factory(
                resource_name, exp_name, config, driver_name
            )
        if len(resources.keys()) > 1:
            raise NotImplementedError(
                "You indicated more than one resource: {list(config['resources'].keys())}"
                " Currently QUEENS only supports one!"
            )
        return resources
    # no specified resources
    else:
        raise Exception("Resources are not properly specified")


def resource_factory(resource_name, exp_name, config, driver_name):
    """Create a resource object.

    Args:
        resource_name (string): name of resource
        exp_name (string):      name of experiment to be run on resource
        config   (dict):        dictionary with problem description
        driver_name (str): Name of driver that should be used in this job-submission

    Returns:
        resource:  resource object constructed from the resource name,
                   exp_name, and config dict
    """
    # get resource options extract resource info from config
    resource_options = config["resources"][resource_name]
    max_concurrent = resource_options.get('max-concurrent', 1)
    max_finished_jobs = resource_options.get('max-finished-jobs', np.inf)

    scheduler_name = resource_options['scheduler']

    # create scheduler from config
    scheduler = from_config_create_scheduler(
        scheduler_name=scheduler_name, config=config, driver_name=driver_name
    )
    # Create/update singularity image in case of cluster job
    scheduler.pre_run()

    return Resource(resource_name, exp_name, scheduler, max_concurrent, max_finished_jobs)


class Resource(object):
    """class which manages a computing resource.

    Attributes:
        name (string):                The name of the resource
        exp_name (string):            The name of the experiment
        scheduler (scheduler object): The object which submits and polls jobs
        scheduler_class (class type): The class type of scheduler.  This is used
                                      just for printing

        max_concurrent (int):         The maximum number of jobs that can run
                                      concurrently on resource

        max_finished_jobs (int):      The maximum number of jobs that can be
                                      run to completion
    """

    def __init__(self, name, exp_name, scheduler, max_concurrent, max_finished_jobs):
        """Init the resource instance.

        Args:
            name (string):                The name of the resource
            exp_name (string):            The name of the experiment
            scheduler (scheduler object): The object which submits and polls jobs
            max_concurrent (int):         The maximum number of jobs that can run
                                          concurrently on resource

            max_finished_jobs (int):      The maximum number of jobs that can be
                                          run to completion
        """
        self.name = name
        self.scheduler = scheduler
        self.scheduler_class = type(scheduler).__name__  # stored just for printing
        self.max_concurrent = max_concurrent
        self.max_finished_jobs = max_finished_jobs
        self.exp_name = exp_name

        if len(self.exp_name) == 0:
            sys.stdout.write("Warning: resource %s has no tasks assigned " " to it" % self.name)

    def filter_my_jobs(self, jobs):
        """Take a list of jobs and filter those that are on this resource.

        Args:
            jobs (list): List with jobs

        Returns:
            list: List with jobs belonging to this resource
        """
        if jobs:
            try:
                return [job for job in jobs if job['resource'] == self.name]
            except TypeError:
                raise TypeError('The design of the jobs list seems wrong! Abort...')
        return jobs

    def accepting_jobs(self, num_pending_jobs):
        """Check if the resource currently is accepting new jobs.

        Args:
            num_pending_jobs (list): number of pending jobs of this resource

        Returns:
            bool: whether or not resource is accepting jobs
        """
        if num_pending_jobs >= self.max_concurrent:
            return False

        return True

    def is_job_alive(self, job):  # TODO this method does not seem to be called
        """Query if a particular job is alive?

        Args:
            job (dict): jobs to query

        Returns:
            bool: whether or not job is alive
        """
        if job['resource'] != self.name:
            raise Exception("This job does not belong to me!")

        return self.scheduler.alive(job['proc_id'])

    def attempt_dispatch(self, batch, job):
        """Submit a new job using the scheduler of the resource.

        Args:
            batch (string):         Batch number of job
            job (dict):             Job to submit

        Returns:
            int:       Process ID of job
        """
        if job['resource'] != self.name:
            raise Exception("This job does not belong to me!")

        process_id = self.scheduler.submit(job['id'], batch)
        if process_id is not None:
            if process_id != 0:
                sys.stdout.write(
                    'Submitted job %d with %s '
                    '(process id: %d).\n' % (job['id'], self.scheduler_class, process_id)
                )
            elif process_id == 0:
                sys.stdout.write(
                    'Checked job %d for restart and loaded results into database.\n\n' % job['id']
                )
        else:
            sys.stdout.write('Failed to submit job %d.\n' % job['id'])

        return process_id

    def check_job_completion(self, job):
        """Check whether this job is completed using the scheduler.

        Args:
            batch (string):         Batch number of job
            job (dict):             Job to check

        Returns:
            int:       Process ID of job
        """
        if job['resource'] != self.name:
            raise Exception("This job does not belong to me!")

        completed, failed = self.scheduler.check_job_completion(job)

        return completed, failed

    def dispatch_post_post_job(self, batch, job):
        """Submit a new post-post job using the scheduler of the resource.

        Args:
            batch (string):         Batch number of job
            job (dict):             Job to submit

        Returns:
            int:       Process ID of job
        """
        if job['resource'] != self.name:
            raise Exception("This job does not belong to me!")

        self.scheduler.submit_post_post(job['id'], batch)
        sys.stdout.write(
            'Submitted post-post job %d with %s \n' % (job['id'], self.scheduler_class)
        )
