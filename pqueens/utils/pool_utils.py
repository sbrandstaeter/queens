"""Pool utils."""
from pathos.multiprocessing import ProcessingPool as Pool


def create_pool(number_of_workers):
    """Create pathos Pool from number of workers.

    Args:
        number_of_workers (int): Number of parallel evaluations

    Returns:
        pathos multiprocessing pool
    """
    if isinstance(number_of_workers, int) and number_of_workers > 1:
        print(f"Activating parallel evaluation of samples with {number_of_workers} workers.\n")
        pool = Pool(processes=number_of_workers)
    else:
        pool = None
    return pool
