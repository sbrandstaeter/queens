"""Schedulers.

The scheduler package contains a set of scheduler classes which submit
compute jobs either through a job-scheduling software or through a
simple system call.
"""


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
    from pqueens.utils.import_utils import get_module_attribute
    from pqueens.utils.valid_options_utils import get_option

    # import here to avoid issues with circular inclusion
    from .cluster_scheduler import ClusterScheduler
    from .standard_scheduler import StandardScheduler

    scheduler_dict = {
        'standard': StandardScheduler,
        'pbs': ClusterScheduler,
        'slurm': ClusterScheduler,
    }

    # get scheduler options according to chosen scheduler name
    # or without specific naming from input file
    if not scheduler_name:
        scheduler_name = "scheduler"
    scheduler_options = config[scheduler_name]
    if scheduler_options.get("external_python_module"):
        module_path = scheduler_options["external_python_module"]
        module_attribute = scheduler_options.get("scheduler_type")
        scheduler_class = get_module_attribute(module_path, module_attribute)
    else:
        scheduler_class = get_option(scheduler_dict, scheduler_options.get("scheduler_type"))

    return scheduler_class.from_config_create_scheduler(config, scheduler_name, driver_name)
