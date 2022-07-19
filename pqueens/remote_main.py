"""This file contains the high-level control routine for remote computations.

Remote computation means the evaluation of the forward model in a
detached process (from the main QUEENS run). The detached process can
run on the same machine as the main run or on a different, remote
machine (e.g., a computing cluster). Additionally, it may also be
wrapped in a singularity image.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import pqueens.database.database as DB_module
from pqueens.drivers import from_config_create_driver

from pqueens.main import get_config_dict

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
    parser.add_argument("--workdir", help="working directory", type=str)

    args = parser.parse_args(args)
    job_id = args.job_id
    batch = args.batch
    port = args.port
    path_json = args.path_json
    post = args.post
    workdir = args.workdir
    driver_name = args.driver_name

    driver_obj = None
    is_remote = port != "000"
    try:
        # If singularity is called remotely
        if is_remote:
            input_path = Path(path_json).joinpath('temp.json')
            # output_dir is not needed but required in get_config_dict
            output_dir = Path(workdir)
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
        with DB_module.database:
            driver_obj = from_config_create_driver(config, job_id, batch, driver_name, workdir)
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
        _logger.error(f"Queens singularity run failed!")
        try:
            if DB_module.database is None:
                _logger.error(f"Could not connect to the database!")
            elif driver_obj is None:
                _logger.error(f"Driver object could not be created!")
            else:
                driver_obj.finalize_job_in_db()
        except Exception as driver_error:
            _logger.error(f"The driver cannot finalize the simulation run(s):")
            raise driver_error from singularity_error
        raise singularity_error


# ------------------------------ HELPER FUNCTIONS -----------------------------
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
