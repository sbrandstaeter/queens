"""QUEENS scheduler parent class."""
import abc
import logging
import subprocess

import numpy as np
import tqdm
from dask.distributed import as_completed

from pqueens.utils.injector import read_file

_logger = logging.getLogger(__name__)

SHUTDOWN_CLIENTS = []


class Scheduler(metaclass=abc.ABCMeta):
    """Abstract base class for schedulers in QUEENS.

    Attributes:
        experiment_name (str): name of QUEENS experiment.
        experiment_dir (Path): Path to QUEENS experiment directory.
        client (Client): Dask client that connects to and submits computation to a Dask cluster
        num_procs (int): number of cores per job
        num_procs_post (int): number of cores per job for post-processing
        restart_worker (bool): If true, restart worker after each finished job
    """

    def __init__(
        self,
        experiment_name,
        experiment_dir,
        client,
        num_procs,
        num_procs_post,
        restart_worker=False,
    ):
        """Initialize scheduler.

        Args:
            experiment_name (str): name of QUEENS experiment.
            experiment_dir (Path): Path to QUEENS experiment directory.
            client (Client): Dask client that connects to and submits computation to a Dask cluster
            num_procs (int): number of cores per job
            num_procs_post (int): number of cores per job for post-processing
            restart_worker (bool): If true, restart worker after each finished job
        """
        self.experiment_name = experiment_name
        self.experiment_dir = experiment_dir
        self.num_procs = num_procs
        self.num_procs_post = num_procs_post
        self.client = client
        self.restart_worker = restart_worker
        global SHUTDOWN_CLIENTS  # pylint: disable=global-variable-not-assigned
        SHUTDOWN_CLIENTS.append(client.shutdown)

    def evaluate(self, samples_list, driver):
        """Submit jobs to driver.

        Args:
            samples_list (list): List of dicts containing samples and job ids
            driver (Driver): Driver object that runs simulation

        Returns:
            result_dict (dict): Dictionary containing results
        """
        futures = self.client.map(
            driver.run,
            samples_list,
            pure=False,
            num_procs=self.num_procs,
            num_procs_post=self.num_procs_post,
            experiment_dir=self.experiment_dir,
            experiment_name=self.experiment_name,
        )

        results = {future.key: None for future in futures}
        with tqdm.tqdm(total=len(futures)) as progressbar:
            for future in as_completed(futures):
                if self.restart_worker:
                    worker = list(self.client.who_has(future).values())[0]
                    results[future.key] = future.result()
                    job_id = self.client.run(
                        lambda: subprocess.check_output('echo $SLURM_JOB_ID', shell=True),
                        workers=list(worker),
                    )
                    job_id = str(list(job_id.values())[0])[2:-3]
                    # _logger.debug(f'scancel %s', job_id)
                    self.client.run_on_scheduler(
                        lambda: subprocess.run(f'scancel {job_id}', check=False, shell=True)
                    )
                progressbar.update(1)

        result_dict = {'mean': [], 'gradient': []}
        for result in results.values():
            # We should remove this squeeze! It is only introduced for consistency with old test.
            result_dict['mean'].append(np.atleast_1d(np.array(result[0]).squeeze()))
            result_dict['gradient'].append(result[1])
        result_dict['mean'] = np.array(result_dict['mean'])
        result_dict['gradient'] = np.array(result_dict['gradient'])
        return result_dict

    def copy_file(self, file_path):
        """Copy file to experiment directory.

        Args:
            file_path (Path): path to file that should be copied to experiment directory
        """
        file = read_file(file_path)
        destination = self.experiment_dir / file_path.name
        self.client.submit(destination.write_text, file, encoding='utf-8').result()

    async def shutdown_client(self):
        """Shutdown the DASK client."""
        await self.client.shutdown()
