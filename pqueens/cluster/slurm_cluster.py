"""Package managing DASK SLURMCluster."""
import argparse
import asyncio
import getpass
import logging
import sys
import time

from dask_jobqueue import SLURMCluster

_logger = logging.getLogger(__name__)

username = getpass.getuser()

DEFAULT_SCHEDULER_PORT = 44444
DEFAULT_DASHBOARD_PORT = 18532
DEFAULT_NUMBER_OF_WORKERS = 1
DEFAULT_CORES_PER_WORKER = 1


async def run_cluster(scheduler_port, dashboard_port, cores_per_worker, num_workers):
    """Start a SLURMCluster."""
    cluster = SLURMCluster(
        cores=cores_per_worker,
        n_workers=num_workers,
        scheduler_options={"port": f"{scheduler_port}", "dashboard_address": f":{dashboard_port}"},
    )

    _logger.debug(cluster)


if __name__ == "__main__":
    args = sys.argv[1:]
    parser = argparse.ArgumentParser(description="QUEENS")
    parser.add_argument(
        '--scheduler-port', type=int, default=DEFAULT_SCHEDULER_PORT, help='Port of scheduler'
    )
    parser.add_argument(
        '--dashboard-port', type=int, default=DEFAULT_DASHBOARD_PORT, help='Port of dashboard'
    )
    parser.add_argument(
        '--cores-per-worker',
        type=int,
        default=DEFAULT_CORES_PER_WORKER,
        help='Number of cores per worker',
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=DEFAULT_NUMBER_OF_WORKERS,
        help='Number of workers on the cluster',
    )
    args = parser.parse_args(args)

    loop = asyncio.get_event_loop()
    try:
        asyncio.ensure_future(
            run_cluster(
                args.scheduler_port, args.dashboard_port, args.cores_per_worker, args.num_workers
            )
        )
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print("Closing Loop")
        loop.close()
