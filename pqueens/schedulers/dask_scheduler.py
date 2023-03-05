"""QUEENS scheduler parent class."""
import abc
import copy
import logging

import numpy as np

_logger = logging.getLogger(__name__)


class Scheduler(metaclass=abc.ABCMeta):
    """Abstract base class for schedulers in QUEENS.

    Attributes:
    """

    def __init__(
        self,
        experiment_name,
        experiment_dir,
        client,
        num_procs,
        num_procs_post,
    ):
        """Initialize scheduler.

        Args:
            experiment_name (str):     name of QUEENS experiment
            experiment_dir (path):     path to QUEENS experiment directory
            client
            num_procs
            num_procs_post
        """
        self.experiment_name = experiment_name
        self.experiment_dir = experiment_dir
        self.num_procs = num_procs
        self.num_procs_post = num_procs_post
        self.client = client

    def evaluate(self, samples_list, driver):
        """Submit job to driver.

        Args:
            samples_list
            driver

        Returns:
            result_dict
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
        results = self.client.gather(futures)

        result_dict = {'mean': [], 'gradient': []}
        for result in results:
            result_dict['mean'].append(result[0])
            result_dict['gradient'].append(result[1])
        result_dict['mean'] = np.array(result_dict['mean'])
        result_dict['gradient'] = np.array(result_dict['gradient'])
        return result_dict

    @staticmethod
    def driver_run(sample_dict, driver, num_procs, num_procs_post, experiment_dir, experiment_name):
        # TODO: This copy is currently necessary because processed data is stored and extended in
        #  data processor
        driver = copy.deepcopy(driver)
        return driver.run(sample_dict, num_procs, num_procs_post, experiment_dir, experiment_name)
