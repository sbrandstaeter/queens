
import sys
import time
import argparse

from pqueens.drivers.baci_driver_docker import baci_driver_docker
from pqueens.drivers.baci_driver_native import baci_driver_native
from pqueens.drivers.python_driver_vector_interface import python_driver_vector_interface
from pqueens.database.mongodb import MongoDB

def main():
    parser = argparse.ArgumentParser(description="QUEENS")

    parser.add_argument('--experiment_name', type=str,
                        help='The name of the experiment in the database.')
    parser.add_argument('--db_address', type=str,
                        help='The address where the database is located.')
    parser.add_argument('--job_id', type=int,
                        help='The id number of the job to launch in the database.')
    parser.add_argument('--batch', type=str,
                        help='The batch to launch in the database.')

    args = parser.parse_args()

    if not args.experiment_name:
        parser.error('Experiment name must be given.')

    if not args.batch:
        parser.error('Batch must be given.')

    if not args.db_address:
        parser.error('Database address must be given.')

    if not args.job_id:
        parser.error('Job ID not given or an ID of 0 was used.')

    launch(args.db_address, args.experiment_name, args.batch, args.job_id)


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
    main()
