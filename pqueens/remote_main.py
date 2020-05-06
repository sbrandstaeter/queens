"""
This file contains the high-level control routine for remote computations.

Remote computation means the evaluation of the forward model in a detached process (from the main
QUEENS run). The detached process can run on the same machine as the main run or on a different,
remote machine (e.g., a computing cluster). Additionally, it may also be wrapped in a singularity
image.
"""

import argparse
from collections import OrderedDict
import os

try:
    import simplejson as json
except ImportError:
    import json
import sys

from pqueens.drivers.driver import Driver


def main(args):
    """
    Main function for remote forward model evaluation.

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
    parser.add_argument("--workdir", help="working directory", type=str)

    args = parser.parse_args(args)
    job_id = args.job_id
    batch = args.batch
    port = args.port
    path_json = args.path_json
    post = args.post
    workdir = args.workdir

    if port == "000":
        try:
            with open(path_json, 'r') as myfile:
                config = json.load(myfile, object_pairs_hook=OrderedDict)
        except FileNotFoundError:
            raise FileNotFoundError("temp.json did not load properly.")

        driver_obj = Driver.from_config_create_driver(config, job_id, batch)

    else:
        try:
            abs_path = os.path.join(path_json, 'temp.json')
            with open(abs_path, 'r') as myfile:
                config = json.load(myfile, object_pairs_hook=OrderedDict)
        except FileNotFoundError:
            raise FileNotFoundError("temp.json did not load properly.")

        path_to_post_post_file = os.path.join(path_json, 'post_post/post_post.py')
        driver_obj = Driver.from_config_create_driver(
            config, job_id, batch, port, path_to_post_post_file, workdir
        )

    # Run the singularity image in two steps
    if post == 'true':
        driver_obj.finish_and_clean()
    else:
        driver_obj.main_run()


# ------------------------------ HELPER FUNCTIONS -----------------------------
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
