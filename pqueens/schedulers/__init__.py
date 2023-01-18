"""Schedulers.

The scheduler package contains a set of scheduler classes which submit
compute jobs either through a job-scheduling software or through a
system call.
"""
import logging

from pqueens.utils.import_utils import get_module_class

_logger = logging.getLogger(__name__)

VALID_TYPES = {
    'standard': ['pqueens.schedulers.standard_scheduler', 'StandardScheduler'],
    'cluster': ['pqueens.schedulers.cluster_scheduler', 'ClusterScheduler'],
}


def from_config_create_scheduler(config, scheduler_name=None, driver_name=None):
    """Create scheduler from problem configuration.

    Args:
        config (dict):        Dictionary containing configuration
                              as provided in the QUEENS input file
        scheduler_name (str): Name of scheduler
        driver_name (str): Name of driver that should be used in this job-submission

    Returns:
        Scheduler object
    """
    # get scheduler options according to chosen scheduler name
    # or without specific naming from input file
    if not scheduler_name:
        scheduler_name = "scheduler"
    scheduler_options = config[scheduler_name]
    scheduler_class = get_module_class(scheduler_options, VALID_TYPES, "type")
    scheduler = scheduler_class.from_config_create_scheduler(config, scheduler_name, driver_name)
    _logger.info(scheduler)
    return scheduler
