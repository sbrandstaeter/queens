"""This file contains the high-level control routine for remote computations.

Remote computation means the evaluation of the forward model in a
detached process (from the main QUEENS run). The detached process can
run on the same machine as the main run or on a different, remote
machine (e.g., a computing cluster). Additionally, it may also be
wrapped in a singularity image.
"""

import argparse
import logging
import sys
from pathlib import Path

import pqueens.database.database as DB_module
from pqueens.drivers import from_config_create_driver
from pqueens.main import get_config_dict
from pqueens.utils.logger_settings import setup_cluster_logging

_logger = logging.getLogger(__name__)


def main(args):
    """Main function for remote forward model evaluation.

    Control routine for the forward model evaluation in an independent, detached process, i.e.,
    remote.
    Called twice per model evaluation: once for the pre-processing and solving and once for the
    post-processing.

    Args:
        args (list): list of arguments to be parsed
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job_id",
        help="specify the job_id so that correct simulation settings can be loaded from the DB",
        type=int,
    )
    parser.add_argument("--batch", help="specify the batch number of the simulation", type=int)
    parser.add_argument("--port", help="port number chosen for port-forwarding", type=str)
    parser.add_argument("--path_json", help="system path to temporary json file", type=str)
    parser.add_argument("--post", help="option for postprocessing", type=str)
    parser.add_argument("--driver_name", help="name of driver for the current run", type=str)
    parser.add_argument("--experiment_dir", help="working directory", type=str)

    args = parser.parse_args(args)
    job_id = args.job_id
    batch = args.batch
    port = args.port
    path_json = Path(args.path_json)
    post = args.post
    experiment_dir = Path(args.experiment_dir)
    driver_name = args.driver_name

    driver_obj = None
    is_remote = port != "000"
    try:
        # If singularity is called remotely
        if is_remote:
            setup_cluster_logging()
            input_path = path_json / 'temp.json'
            # output_dir is not needed but required in get_config_dict
            output_dir = path_json
            config = get_config_dict(input_path, output_dir)

            # Patch the remote address to the config
            remote_address = (
                str(config["scheduler"]["singularity_settings"]["remote_ip"]) + ":" + str(port)
            )
            config["database"]["address"] = remote_address
        else:
            input_path = Path(path_json)
            # output_dir is not needed but required in get_config_dict
            output_dir = input_path.parent
            config = get_config_dict(input_path, output_dir)

        # Do not delete existing db
        config["database"]["reset_existing_db"] = False

        # Create database
        DB_module.from_config_create_database(config)
        with DB_module.database:  # pylint: disable=no-member
            driver_obj = from_config_create_driver(
                config=config,
                job_id=job_id,
                batch=batch,
                driver_name=driver_name,
                experiment_dir=experiment_dir,
            )
            # Run the singularity image in two steps and two different singularity calls to have
            # more freedom concerning mpi ranks
            if is_remote:
                if post == 'true':
                    driver_obj.post_job_run()
                else:
                    driver_obj.pre_job_run_and_run_job()
            else:
                driver_obj.pre_job_run_and_run_job()
                driver_obj.post_job_run()

    except Exception as singularity_error:
        _logger.error("Queens singularity run failed!")
        try:
            if DB_module.database is None:  # pylint: disable=no-member
                _logger.error("Could not connect to the database!")
            elif driver_obj is None:
                _logger.error("Driver object could not be created!")
            else:
                driver_obj.finalize_job_in_db()
        except Exception as driver_error:
            _logger.error("The driver cannot finalize the simulation run(s):")
            raise driver_error from singularity_error
        raise singularity_error


# ------------------------------ HELPER FUNCTIONS -----------------------------
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
