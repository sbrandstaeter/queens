"""QUEENS scheduler parent class."""

import abc
import logging
import time

import numpy as np
import tqdm
from dask.distributed import as_completed

_logger = logging.getLogger(__name__)

SHUTDOWN_CLIENTS = []


class Scheduler(metaclass=abc.ABCMeta):
    """Abstract base class for schedulers in QUEENS.

    Attributes:
        experiment_name (str): name of the current experiment
        experiment_dir (Path): Path to QUEENS experiment directory.
        client (Client): Dask client that connects to and submits computation to a Dask cluster
        num_procs (int): number of processors per job
        restart_workers (bool): If true, restart workers after each finished job
    """

    def __init__(
        self,
        experiment_name,
        experiment_dir,
        client,
        num_procs,
        restart_workers,
    ):
        """Initialize scheduler.

        Args:
            experiment_name (str): name of QUEENS experiment.
            experiment_dir (Path): Path to QUEENS experiment directory.
            client (Client): Dask client that connects to and submits computation to a Dask cluster
            num_procs (int): number of processors per job
            restart_workers (bool): If true, restart workers after each finished job
        """
        self.experiment_name = experiment_name
        self.experiment_dir = experiment_dir
        self.num_procs = num_procs
        self.client = client
        self.restart_workers = restart_workers
        global SHUTDOWN_CLIENTS  # pylint: disable=global-variable-not-assigned
        SHUTDOWN_CLIENTS.append(client.shutdown)

    def evaluate(self, job_ids_and_samples, driver):
        """Submit jobs to driver.

        Args:
            job_ids_and_samples (np.array): array containing job ids and samples (row-wise)
            driver (Driver): Driver object that runs simulation

        Returns:
            result_dict (dict): Dictionary containing results
        """
        if self.restart_workers:
            # This is necessary, because the subprocess in the driver does not get killed
            # sometimes when the worker is restarted.
            def run_driver(*args, **kwargs):
                time.sleep(5)
                return driver.run(*args, **kwargs)

        else:
            run_driver = driver.run

        futures = self.client.map(
            run_driver,
            job_ids_and_samples,
            pure=False,
            num_procs=self.num_procs,
            experiment_dir=self.experiment_dir,
            experiment_name=self.experiment_name,
        )

        results = {future.key: None for future in futures}
        with tqdm.tqdm(total=len(futures)) as progressbar:
            for future in as_completed(futures):
                results[future.key] = future.result()
                progressbar.update(1)
                if self.restart_workers:
                    worker = list(self.client.who_has(future).values())[0]
                    self.restart_worker(worker)

        result_dict = {"result": [], "gradient": []}
        for result in results.values():
            # We should remove this squeeze! It is only introduced for consistency with old test.
            result_dict["result"].append(np.atleast_1d(np.array(result[0]).squeeze()))
            result_dict["gradient"].append(result[1])
        result_dict["result"] = np.array(result_dict["result"])
        result_dict["gradient"] = np.array(result_dict["gradient"])
        return result_dict

    @abc.abstractmethod
    def copy_files_to_experiment_dir(self, paths):
        """Copy file to experiment directory."""

    @abc.abstractmethod
    def restart_worker(self, worker):
        """Restart a worker."""

    async def shutdown_client(self):
        """Shutdown the DASK client."""
        await self.client.shutdown()
