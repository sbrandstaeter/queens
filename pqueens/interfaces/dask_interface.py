"""Job interface class."""
import atexit
import logging
import time

import numpy as np
from dask.distributed import Client, LocalCluster, as_completed

from pqueens.cluster.manage_cluster import RemoteClusterManager
from pqueens.drivers.dask_driver import DaskDriver
from pqueens.interfaces.interface import Interface
from pqueens.utils.config_directories import experiment_directory

_logger = logging.getLogger(__name__)


class DaskInterface(Interface):
    """Class for mapping input variables to responses.

    The *JobInterface* class maps input variables to outputs, i.e. responses
    by creating a job which is then submitted to a job manager on some
    local or remote resource, which in turn then actually runs the
    simulation software.

    Attributes:
        name (string): Name of interface.
        experiment_name (string): Name of experiment.
        output_dir (path): Directory to write output to.
        batch_number (int): Number of the current simulation batch.
        driver_name (str): Name of the associated driver for the current interface.
        _internal_batch_state (int): Helper attribute to compare *batch_number* with the internal
        experiment_dir (path):                       directory to write output to
        job_num (int):              Number of the current job
        _internal_batch_state (int): Helper attribute to compare batch_number with the internal
    """

    def __init__(
        self,
        interface_name,
        experiment_name,
        experiment_dir,
        driver_name,
        config,
        remote,
        remote_connect,
    ):
        """Create JobInterface.

        Args:
            interface_name (string):    name of interface
            experiment_name (string):   name of experiment
            experiment_dir (path):          directory to write output to
            driver_name (str):          Name of the associated driver for the current interface
            remote (bool):              indicate if dask cluster runs locally or remote
        """
        super().__init__(interface_name)
        self.name = interface_name
        self.experiment_name = experiment_name
        self.experiment_dir = experiment_dir
        self.batch_number = 0
        self.driver_name = driver_name
        self._internal_batch_state = 0
        self.job_num = 0
        self.remote = remote

        driver_options = config[driver_name]
        num_procs = driver_options.get('num_procs', 1)
        num_procs_post = driver_options.get('num_procs_post', 1)
        cores_per_worker = max(num_procs, num_procs_post)

        num_workers = config[interface_name].get("num_workers", 1)

        if self.remote:
            scheduler_port = config[interface_name].get("scheduler_port", 44444)
            dashboard_port = config[interface_name].get("dashboard_port", 8787)
            scheduler_address = config[interface_name].get("scheduler_address")
            if scheduler_address is None:
                raise ValueError("You need to provide the IP address the scheduler will run on.")

            self.remote_cluster_manager = RemoteClusterManager(
                remote_connect=remote_connect,
                scheduler_port=scheduler_port,
                scheduler_address=scheduler_address,
                cores_per_worker=cores_per_worker,
                num_workers=num_workers,
                dashboard_port=dashboard_port,
            )
            self.remote_cluster_manager.setup_cluster()
            self.client = Client(address=f"localhost:{scheduler_port}")
        else:
            cluster = LocalCluster(n_workers=num_workers, threads_per_worker=cores_per_worker)
            self.client = Client(cluster)

        _logger.info(self.client)
        _logger.info(self.client.dashboard_link)

        self.config = config

        self.driver_obj = DaskDriver.from_config_create_driver(
            config=self.config,
            job_id=0,
            batch=0,
            driver_name=self.driver_name,
            experiment_dir=self.experiment_dir,
            initial_working_dir=None,
            job={},
        )

        atexit.register(self.shutdown_dask)

    @classmethod
    def from_config_create_interface(cls, interface_name, config):
        """Create JobInterface from config dictionary.

        Args:
            interface_name (str):   Name of interface.
            config (dict):          Dictionary containing problem description.

        Returns:
            interface: Instance of JobInterface
        """
        # get experiment name and polling time
        experiment_name = config['global_settings']['experiment_name']

        interface_options = config[interface_name]
        driver_name = interface_options.get('driver_name', None)
        if driver_name is None:
            raise ValueError("No driver_name specified for the JobInterface.")

        # get flag for remote scheduling
        if interface_options.get('remote', False):
            remote = True
            remote_connect = interface_options['remote_connect']
        else:
            remote_connect = None
            remote = False

        experiment_dir = experiment_directory(
            experiment_name=experiment_name, remote_connect=remote_connect
        )

        # instantiate object
        return cls(
            interface_name,
            experiment_name,
            experiment_dir,
            driver_name,
            config,
            remote,
            remote_connect,
        )

    def shutdown_dask(self):
        """Collect all methods needed to shut down the client and cluster."""
        self.client.close()
        time.sleep(0.5)
        self.remote_cluster_manager.shutdown_cluster()

    def evaluate(self, samples, gradient_bool=False):
        """Orchestrate call to external simulation software.

        Second variant which takes the input samples as argument.

        Args:
            samples (np.ndarray): Realization/samples of QUEENS simulation input variables
            gradient_bool (bool): Flag to determine whether the gradient of the function at
                                  the evaluation point is expected (*True*) or not (*False*)

        Returns:
            output (dict): Output data
        """
        self.batch_number += 1

        # Main run
        _logger.info("Executing %s jobs.\n", samples.shape[0])
        jobs = self._manage_jobs(samples)

        # get sample and response data
        output = self.get_output_data(
            num_samples=samples.shape[0], gradient_bool=gradient_bool, jobs=jobs
        )
        return output

    def create_new_job(self, variables, resource_name, new_id=None):
        """Create new job and save it to database and return it.

        Args:
            variables (Variables):     Variables to run model at
            resource_name (string):     Name of resource
            new_id (int):                  ID for job

        Returns:
            job (dict): New job
        """
        job_id = int(new_id)

        job = {
            'id': job_id,
            'params': variables,
            'experiment_dir': str(self.experiment_dir),
            'experiment_name': self.experiment_name,
            'resource': resource_name,
            'status': "",
            'submit_time': time.time(),
            'start_time': 0.0,
            'end_time': 0.0,
            'driver_name': self.driver_name,
        }

        return job

    def get_output_data(self, num_samples, gradient_bool, jobs):
        """Extract output data from database and return it.

        Args:
            num_samples (int): Number of evaluated samples
            gradient_bool (bool): Flag to determine whether the gradient
                                  of the model output w.r.t. the input
                                  is expected (*True* if yes)

        Returns:
            dict: Output dictionary; i
                +------------+------------------------------------------------+
                |**key:**    |**value:**                                      |
                +------------+------------------------------------------------+
                |'mean'      | ndarray shape(batch_size, shape_of_response)   |
                +------------+------------------------------------------------+
                | 'var'      | ndarray (optional)                             |
                +------------+------------------------------------------------+
        """
        output = {}
        mean_values = []
        gradient_values = []
        # Sort job IDs in ascending order to match ordering of samples
        jobids = [job['id'] for job in jobs]
        jobids.sort()

        for current_job_id in jobids:
            current_job = next(job for job in jobs if job['id'] == current_job_id)
            mean_value = np.squeeze(current_job['result'])
            gradient_value = np.squeeze(current_job.get('gradient', None))

            if not mean_value.shape:
                mean_value = np.expand_dims(mean_value, axis=0)
                gradient_value = np.expand_dims(gradient_value, axis=0)

            mean_values.append(mean_value)
            gradient_values.append(gradient_value)

        output['mean'] = np.array(mean_values)[-num_samples:]
        if gradient_bool:
            output['gradient'] = np.array(gradient_values)[-num_samples:]

        return output

    # ------------ private helper methods -------------- #

    def _manage_jobs(self, samples):
        """Manage regular submission of jobs.

        Args:
            samples (DataFrame): realization/samples of QUEENS simulation input variables

        Returns:
            jobid_for_data_processor(ndarray): jobids for data-processing
        """
        num_jobs = 0
        if not num_jobs or self.batch_number == 1:
            job_ids_generator = range(1, samples.shape[0] + 1, 1)
        else:
            job_ids_generator = range(num_jobs + 1, num_jobs + samples.shape[0] + 1, 1)

        jobs = self._manage_job_submission(samples, job_ids_generator)

        return jobs

    def _manage_job_submission(self, samples, jobid_range):
        """Iterate over samples and manage submission of jobs.

        Args:
            samples (DataFrame):     realization/samples of QUEENS simulation input variables
            jobid_range (range):     range of job IDs which are submitted
        """
        futures = []
        for jobid in jobid_range:
            if self._internal_batch_state != self.batch_number:
                self._internal_batch_state = self.batch_number
                self.job_num = 0

            self.job_num += 1
            sample_dict = self.parameters.sample_as_dict(samples[self.job_num - 1])
            resource_name = "DASK"
            current_job = self.create_new_job(sample_dict, resource_name, jobid)

            current_job['status'] = 'pending'

            futures.append(
                self.client.submit(
                    self.execute_driver,
                    self.driver_obj,
                    jobid,
                    self.batch_number,
                    current_job,
                    key=f"job-{jobid}-batch-{self.batch_number}",
                )
            )

        completed_futures = as_completed(futures)

        jobs = []
        for completed_future in completed_futures:
            jobs.append(completed_future.result())

        return jobs

    @staticmethod
    def execute_driver(driver_obj, job_id, batch, job):
        """Help execute driver.

        Args:
            driver_obj (MPIDriver): MPIDriver executing the forward solver
            job_id (int): ID number of the current job
            batch (int): Number of the current batch
            job (dict): contains all data of the job including variable values

        Returns:
            pid (int): Process ID
        """
        # run driver and get result (in job dict)
        driver_obj.set_job(job_id, batch, job)
        driver_obj.pre_job_run_and_run_job()
        driver_obj.post_job_run()
        job = driver_obj.job
        return job
