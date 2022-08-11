"""Schedulers.

The scheduler package contains a set of scheduler classes which submit
compute jobs either through a job-scheduling software or through a
simple system call.
"""
from pqueens.utils.import_utils import get_module_class


def from_config_create_scheduler(config, scheduler_name=None, driver_name=None):
    """Create scheduler from problem configuration.

    Args:
        config (dict):        dictionary containing configuration
                              as provided in QUEENS input file
        scheduler_name (str): Name of scheduler
        driver_name (str): Name of driver that should be used in this job-submission

    Returns:
        Scheduler object
    """
    valid_types = {
        'standard': ['.standard_scheduler', 'StandardScheduler'],
        'pbs': ['.cluster_scheduler', 'ClusterScheduler'],
        'slurm': ['.cluster_scheduler', 'ClusterScheduler'],
    }

    # get scheduler options according to chosen scheduler name
    # or without specific naming from input file
    if not scheduler_name:
        scheduler_name = "scheduler"
    scheduler_options = config[scheduler_name]
    scheduler_type = scheduler_options.get("scheduler_type")
    scheduler_class = get_module_class(scheduler_options, valid_types, scheduler_type)

    return scheduler_class.from_config_create_scheduler(config, scheduler_name, driver_name)
