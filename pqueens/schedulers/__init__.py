"""Schedulers.

The scheduler package contains a set of scheduler classes which submit
compute jobs either through a job-scheduling software or through a
system call.
"""
import logging

from pqueens.utils.import_utils import get_module_class

_logger = logging.getLogger(__name__)

VALID_TYPES = {
    'local': ['pqueens.schedulers.local_scheduler', 'LocalScheduler'],
    'cluster': ['pqueens.schedulers.cluster_scheduler', 'ClusterScheduler'],
}


def from_config_create_scheduler(config, scheduler_name):
    """Create scheduler from problem configuration.

    Args:
        config (dict):        Dictionary containing configuration
                              as provided in the QUEENS input file
        scheduler_name (str): Name of scheduler

    Returns:
        Scheduler object
    """
    scheduler_options = config[scheduler_name]
    scheduler_class = get_module_class(scheduler_options, VALID_TYPES, "type")
    scheduler = scheduler_class.from_config_create_scheduler(config, scheduler_name)
    return scheduler
