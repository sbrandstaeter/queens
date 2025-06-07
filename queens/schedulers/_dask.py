#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (c) 2024-2025, QUEENS contributors.
#
# This file is part of QUEENS.
#
# QUEENS is free software: you can redistribute it and/or modify it under the terms of the GNU
# Lesser General Public License as published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version. QUEENS is distributed in the hope that it will
# be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details. You
# should have received a copy of the GNU Lesser General Public License along with QUEENS. If not,
# see <https://www.gnu.org/licenses/>.
#
"""QUEENS dask scheduler parent class."""

import abc
import logging
import time
import random

import numpy as np
import pandas as pd
import tqdm
from dask.distributed import as_completed, progress
from distributed import WorkerPlugin


from queens.schedulers._scheduler import Scheduler
from queens.utils.printing import get_str_table

_logger = logging.getLogger(__name__)

SHUTDOWN_CLIENTS = []


class ShutdownAfterFirstTask(WorkerPlugin):
    def setup(self, worker):
        self.worker = worker
        self.has_shutdown = False

    def transition(self, key, start, finish, *args, **kwargs):

        # if start == "ready" and finish == "executing" and not self.has_shutdown:
        #    self.worker.state = "closing"
        if start == "executing" and finish == "memory" and not self.has_shutdown:
            self.has_shutdown = True

            async def shutdown():
                await self.worker.close_gracefully(reason=f"Shutdown after task {key}")

            self.worker.loop.call_later(2, shutdown)


class Dask(Scheduler):
    """Abstract base class for schedulers in QUEENS.

    Attributes:
        num_procs (int): number of processors per job
        client (Client): Dask client that connects to and submits computation to a Dask cluster
        restart_workers (bool): If true, restart workers after each finished job
    """

    def __init__(
        self,
        experiment_name,
        experiment_dir,
        num_jobs,
        num_procs,
        client,
        restart_workers,
        verbose=True,
    ):
        """Initialize scheduler.

        Args:
            experiment_name (str): name of QUEENS experiment.
            experiment_dir (Path): Path to QUEENS experiment directory.
            num_jobs (int): Maximum number of parallel jobs
            num_procs (int): number of processors per job
            client (Client): Dask client that connects to and submits computation to a Dask cluster
            restart_workers (bool): If true, restart workers after each finished job
            verbose (bool, opt): Verbosity of evaluations. Defaults to True.
        """
        super().__init__(
            experiment_name=experiment_name,
            experiment_dir=experiment_dir,
            num_jobs=num_jobs,
            verbose=verbose,
        )
        self.num_procs = num_procs
        self.client = client
        self.restart_workers = restart_workers

        # Register this plugin on all workers
        if self.restart_workers:
            self.client.register_plugin(ShutdownAfterFirstTask(), name="shutdown_after_one")

        global SHUTDOWN_CLIENTS  # pylint: disable=global-variable-not-assigned
        SHUTDOWN_CLIENTS.append(client.shutdown)

    def evaluate(self, samples, driver, job_ids=None):
        """Submit jobs to driver.

        Args:
            samples (np.array): Array of samples
            driver (Driver): Driver object that runs simulation
            job_ids (lst, opt): List of job IDs corresponding to samples

        Returns:
            result_dict (dict): Dictionary containing results
        """

        # the initial batch can overwhelm the hardware infrastructue by starting many jobs at the same time
        # -> introduce a random wait time for jobs to spread the load
        def run_driver(*args, **kwargs):
            random_wait_time = random.uniform(3, 7)
            time.sleep(random_wait_time)
            return driver.run(*args, **kwargs)

        if job_ids is None:
            job_ids = self.get_job_ids(len(samples))
        futures = self.client.map(
            run_driver,
            samples,
            job_ids,
            pure=False,
            num_procs=self.num_procs,
            experiment_dir=self.experiment_dir,
            experiment_name=self.experiment_name,
        )

        # The theoretical number of sequential jobs
        num_sequential_jobs = int(np.ceil(len(samples) / self.num_jobs))

        start_time = time.time()
        progress(futures)
        results_values = self.client.gather(futures)

        if self.verbose:
            elapsed_time = time.time() - start_time
            averaged_time_per_job = elapsed_time / num_sequential_jobs

            run_time_dict = {
                "number of jobs": len(samples),
                "number of parallel jobs": self.num_jobs,
                "number of procs": self.num_procs,
                "total elapsed time": f"{elapsed_time:.3e}s",
                "average time per parallel job": f"{averaged_time_per_job:.3e}s",
            }
            _logger.info(
                get_str_table(
                    f"Batch summary for jobs {min(job_ids)} - {max(job_ids)}", run_time_dict
                )
            )

        result_dict = {"result": [], "gradient": []}
        # for result in results:
        for result in results_values:
            # We should remove this squeeze! It is only introduced for consistency with old test.
            result_dict["result"].append(np.atleast_1d(np.array(result[0]).squeeze()))
            result_dict["gradient"].append(result[1])
        result_df = pd.DataFrame(result_dict["result"], dtype="float")
        result_dict["result"] = result_df.values
        result_dict["gradient"] = np.array(result_dict["gradient"])
        return result_dict

    @abc.abstractmethod
    def restart_worker(self, worker):
        """Restart a worker."""

    async def shutdown_client(self):
        """Shutdown the DASK client."""
        await self.client.shutdown()
