
#from pqueens.schedulers.scheduler_factory import SchedulerFactory
from pqueens.schedulers.scheduler import Scheduler
#import importlib
from operator import add
from functools import reduce
import numpy as np
import sys

# TODO refactor this method into a class method

def parse_resources_from_configuration(config):
    """ Parse the configuration dictionary

    Args:
        config (dict): Dictionary with problem description
    Returns:
        dict: Dictionary with resource objects keyed with resource name
    """

    if "resources" in config:
        resources = dict()
        for resource_name, _ in config["resources"].items():
            global_settings = config.get("global_settings")
            exp_name = global_settings.get("experiment_name")
            resources[resource_name] = resource_factory(resource_name,
                                                        exp_name,
                                                        config)
        return resources
    # no specified resources
    else:
        raise Exception("Resources are not properly specified")

def resource_factory(resource_name, exp_name, config):
    """ Create a resource object

    Args:
        resource_name (string): name of resource
        exp_name (string):      name of experiment to be run on resource
        config   (dict):        dictionary with problem description

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
    scheduler = Scheduler.from_config_create_scheduler(scheduler_name, config)

    return Resource(resource_name, exp_name, scheduler, max_concurrent,
                    max_finished_jobs)

def print_resources_status(resources, jobs):
    """ Print out whats going on on the resources
    Args:
        resources (dict):   Dictionary with used resources
        jobs (list):        List of jobs

    """
    sys.stderr.write('\nResources:      ')
    left_indent = 16
    indentation = ' '*left_indent
    sys.stderr.write('NAME          PENDING    COMPLETE    FAILED\n')
    sys.stderr.write(indentation)
    sys.stderr.write('----          -------    --------    ------\n')
    total_pending = 0
    total_complete = 0
    total_failed = 0
    #for resource in resources:
    for _, resource in resources.items():
        p = resource.num_pending(jobs)
        c = resource.num_complete(jobs)
        f = resource.num_failed(jobs)
        total_pending += p
        total_complete += c
        total_failed += f
        sys.stderr.write("%s%-12.12s  %-9d  %-10d  %-9d\n" % (indentation,
                                                              resource.name,
                                                              p, c, f))
    sys.stderr.write("%s%-12.12s  %-9d  %-10d  %-9d\n" % (indentation, '*TOTAL*',
                                                          total_pending,
                                                          total_complete,
                                                          total_failed))
    sys.stderr.write('\n')

class Resource(object):
    """class which manages computing resources

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
        """
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
        self.scheduler_class = type(scheduler).__name__   # stored just for printing
        self.max_concurrent = max_concurrent
        self.max_finished_jobs = max_finished_jobs
        self.exp_name = exp_name

        if len(self.exp_name) == 0:
            sys.stderr.write("Warning: resource %s has no tasks assigned "
                             " to it" % self.name)


    def filter_my_jobs(self, jobs):
        """ Take a list of jobs and filter those that are on this resource

        Args:
            jobs (list): List with jobs

        Returns:
            list: List with jobs belonging to this resource

        """
        if jobs:
            return filter(lambda job: job['resource']==self.name, jobs)
        else:
            return jobs

    def num_pending(self, jobs):
        """ Take a list of jobs and filter those that are either pending or new

        Args:
            jobs (list): List with jobs

        Returns:
            list: List with jobs that are either pending or new

        """
        jobs = self.filter_my_jobs(jobs)
        if jobs:
            pending_jobs = map(lambda x: x['status'] in ['pending', 'new'], jobs)
            return reduce(add, pending_jobs, 0)
        else:
            return 0

    def num_failed(self, jobs):
        """ Take a list of jobs and filter those that have failed

        Args:
            jobs (list): List with jobs

        Returns:
            list: List with jobs that have failed

        """
        jobs = self.filter_my_jobs(jobs)
        if jobs:
            failed_jobs = map(lambda x: x['status'] in ['failed'], jobs)
            return reduce(add, failed_jobs, 0)
        else:
            return 0


    def num_complete(self, jobs):
        """ Take a list of jobs and filter those that are complete

        Args:
            jobs (list): List with jobs

        Returns:
            list: List with jobs that either are complete

        """
        jobs = self.filter_my_jobs(jobs)
        if jobs:
            completed_jobs = map(lambda x: x['status'] == 'complete', jobs)
            return reduce(add, completed_jobs, 0)
        else:
            return 0

    def accepting_jobs(self, jobs):
        """ Is this resource currently accepting new jobs?

        Args:
            jobs (list): List with jobs

        Returns:
            bool: whether or not resource is accepting jobs

        """
        if self.num_pending(jobs) >= self.max_concurrent:
            return False

        if self.num_complete(jobs) >= self.max_finished_jobs:
            return False

        return True

    def print_status(self, jobs):
        """ Print number of pending ans completed jobs

        Args:
            jobs (list): List with jobs
        """
        sys.stderr.write("%-12s: %5d pending %5d complete\n" %
                         (self.name, self.num_pending(jobs), self.num_complete(jobs)))

    def is_job_alive(self, job):
        """ Query if a particular job is alive?

        Args:
            job (dict): jobs to query

        Returns:
            bool: whether or not job is alive

        """
        if job['resource'] != self.name:
            raise Exception("This job does not belong to me!")

        return self.scheduler.alive(job['proc_id'])

    def attempt_dispatch(self, experiment_name, batch, job, db_address, expt_dir):
        """ Submit a new job using the scheduler of the resource

        Args:
            experiment_name (str):  Name of experiment
            batch (string):         Batch number of job
            job (dict):             Job to submit
            db_address (str):       Adress of database to store job info in
            expt_dir  (str):        Directory associated with experiment

        Returns:
            int:       Process ID of job
        """
        if job['resource'] != self.name:
            raise Exception("This job does not belong to me!")

        process_id = self.scheduler.submit(job['id'], experiment_name, batch,
                                           expt_dir, db_address, job['driver_params'])

        if process_id is not None:
            sys.stderr.write('Submitted job %d with %s '
                             '(process id: %d).\n' %
                             (job['id'], self.scheduler_class, process_id))
        else:
            sys.stderr.write('Failed to submit job %d.\n' % job['id'])

        return process_id
