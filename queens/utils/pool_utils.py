"""Pool utils."""

import logging

from pathos.multiprocessing import ProcessingPool as Pool

_logger = logging.getLogger(__name__)


def create_pool(number_of_workers):
    """Create pathos Pool from number of workers.

    Args:
        number_of_workers (int): Number of parallel evaluations

    Returns:
        pathos multiprocessing pool
    """
    if isinstance(number_of_workers, int) and number_of_workers > 1:
        _logger.info(
            "Activating parallel evaluation of samples with %s workers.\n", number_of_workers
        )
        pool = Pool(processes=number_of_workers)
    else:
        pool = None
    return pool
