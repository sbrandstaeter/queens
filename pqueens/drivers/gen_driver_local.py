import sys
import time
import argparse

from pqueens.drivers.baci_driver_docker import baci_driver_docker
from pqueens.drivers.fenics_driver_docker_new import fenics_driver_docker
from pqueens.drivers.baci_driver_native import baci_driver_native
from pqueens.drivers.ansys_driver_native import ansys_driver_native
from pqueens.drivers.python_driver_vector_interface import python_driver_vector_interface
from pqueens.database.mongodb import MongoDB

def main(args):
    """ Launch driver

        Args:
            args (list): Command line options
    """
    parsed_args = parse_args(args)
    launch(parsed_args.db_address, parsed_args.experiment_name,
           parsed_args.batch, parsed_args.job_id)

def parse_args(args):
    """ Parse command line options

        Args:
            args: Commad line options

        Returns:
            dict: Parsed command line options
    """
    parser = argparse.ArgumentParser(description="QUEENS")

    parser.add_argument('--experiment_name', type=str,
                        help='The name of the experiment in the database.')
    parser.add_argument('--db_address', type=str,
                        help='The address where the database is located.')
    parser.add_argument('--job_id', type=int,
                        help='The id number of the job to launch in the database.')
    parser.add_argument('--batch', type=str,
                        help='The batch to launch in the database.')

    parsed_args = parser.parse_args(args)

    if not parsed_args.experiment_name:
        raise RuntimeError('Experiment name must be given.')

    if not parsed_args.db_address:
        raise RuntimeError('Database address must be given.')

    if not parsed_args.job_id:
        raise RuntimeError('Job ID not given or an ID of 0 was used.')

    if not parsed_args.batch:
        raise RuntimeError('Batch must be given.')

    return parsed_args

def launch(db_address, experiment_name, batch, job_id):
    """
    Launches a job from based on a given id.

    Args:
        db_address (string):        Adress of the database
        experiment_name (string):   Name of experiment_name
        batch (string):             Batch number the job runs in
        job_id (int):               ID of job to run

    """

    db  = MongoDB(database_address = db_address)
    job = db.load(experiment_name, batch, 'jobs', {'id' : job_id})
    print(job)
    start_time = time.time()
    print("start_time {}".format(start_time))
    job['start time'] = start_time
    db.save(job, experiment_name, 'jobs', batch, {'id' : job_id})

    sys.stderr.write("Job launching after %0.2f seconds in submission.\n"
                     % (start_time-job['submit time']))

    success = False

    try:
        if job['driver_type'].lower() == 'python_vector_interface':
            result = python_driver_vector_interface(job)
        elif job['driver_type'].lower() == 'baci_docker':
            result = baci_driver_docker(job)
        elif job['driver_type'].lower() == 'baci_native':
            result = baci_driver_native(job)
        elif job['driver_type'].lower() == 'fenics_docker':
            result = fenics_driver_docker(job)
        elif job['driver_type'].lower() == 'ansys_native':
            result = ansys_driver_native(job)
        else:
            raise Exception("That driver type has not been implemented.")

        success = True
    except:
        import traceback
        traceback.print_exc()
        sys.stderr.write("Problem executing the function\n")
        print(sys.exc_info())

    end_time = time.time()

    if success:
        sys.stderr.write("Completed successfully in %0.2f seconds. [%s]\n"
                         % (end_time-start_time, result))

        job['result'] = result
        job['status'] = 'complete'
        job['end time'] = end_time

    else:
        sys.stderr.write("Job failed in %0.2f seconds.\n" % (end_time-start_time))

        # Update metadata.
        job['status'] = 'broken'
        job['end time'] = end_time

    db.save(job, experiment_name, 'jobs', batch, {'id' : job_id})

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
