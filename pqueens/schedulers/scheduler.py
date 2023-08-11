"""QUEENS scheduler parent class."""
import abc
import copy
import logging

import numpy as np
from dask.distributed import progress

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
        progressbar (bool): If true, print progressbar. WARNING: If multiple dask schedulers are
                            used, the progressbar must be disabled.
    """

    def __init__(
        self,
        experiment_name,
        experiment_dir,
        client,
        num_procs,
        num_procs_post,
        progressbar,
    ):
        """Initialize scheduler.

        Args:
            experiment_name (str): name of QUEENS experiment.
            experiment_dir (Path): Path to QUEENS experiment directory.
            client (Client): Dask client that connects to and submits computation to a Dask cluster
            num_procs (int): number of cores per job
            num_procs_post (int): number of cores per job for post-processing
            progressbar (bool): If true, print progressbar. WARNING: If multiple dask schedulers are
                                used, the progressbar must be disabled.
        """
        self.experiment_name = experiment_name
        self.experiment_dir = experiment_dir
        self.num_procs = num_procs
        self.num_procs_post = num_procs_post
        self.client = client
        self.progressbar = progressbar
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
            self.driver_run,
            samples_list,
            pure=False,
            driver=driver,
            num_procs=self.num_procs,
            num_procs_post=self.num_procs_post,
            experiment_dir=self.experiment_dir,
            experiment_name=self.experiment_name,
        )
        if self.progressbar:
            progress(futures)
        results = self.client.gather(futures)

        result_dict = {'mean': [], 'gradient': []}
        for result in results:
            # We should remove this squeeze! It is only introduced for consistency with old test.
            result_dict['mean'].append(np.atleast_1d(np.array(result[0]).squeeze()))
            result_dict['gradient'].append(result[1])
        result_dict['mean'] = np.array(result_dict['mean'])
        result_dict['gradient'] = np.array(result_dict['gradient'])
        return result_dict

    @staticmethod
    def driver_run(sample_dict, driver, num_procs, num_procs_post, experiment_dir, experiment_name):
        """Run the driver.

        Args:
            sample_dict (list): Dict containing sample and job id
            driver (Driver): Driver object that runs simulation
            num_procs (int): number of cores per job
            num_procs_post (int): number of cores per job for post-processing
            experiment_name (str): name of QUEENS experiment.
            experiment_dir (Path): Path to QUEENS experiment directory.

        Returns:
            Result and potentially the gradient
        """
        #  This copy is currently necessary because processed data is stored and extended in
        #  data processor
        driver = copy.deepcopy(driver)
        return driver.run(sample_dict, num_procs, num_procs_post, experiment_dir, experiment_name)

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
