"""Module supplies functions for printing out information at the beginning of
each QUEENS run."""

import sys


def print_scheduling_information(scheduler_type, remote, remote_connect, singularity):
    """Print out information on chosen scheduling."""
    # determine scheduler name
    scheduler_dict = {
        'standard': 'standard',
        'slurm': 'Slurm',
        'pbs': 'PBS/Torque',
    }
    scheduler_name = scheduler_dict[scheduler_type]

    sys.stdout.write('\n=====================================================================')
    sys.stdout.write('\nScheduling Information:      ')
    sys.stdout.write('\n=====================================================================')
    if not remote:
        sys.stdout.write('\nJobs will be run on local computing resource.')
    else:
        sys.stdout.write(
            '\nJobs will be run on remote computing resource with host name'
            '\n(or IP address):\n\t%s' % remote_connect
        )
    sys.stdout.write('\nChosen type of scheduling:\n\t%s' % scheduler_name)
    if singularity:
        sys.stdout.write('\nAs requested, all jobs will be run in Singularity containers.')
    else:
        if remote:
            sys.stdout.write(
                '\nPlease note that remote scheduling without using Singularity'
                '\ncontainers might result in enhanced network traffic!'
            )
    sys.stdout.write('\n=====================================================================')
    sys.stdout.write('\n')


def print_driver_information(
    driver_type, cae_software_version, post_post_file_prefix, docker_image
):
    """Print out information on chosen driver."""
    # determine name of driver
    driver_dict = {
        'baci': 'BACI',
    }
    driver_name = driver_dict[driver_type]

    sys.stdout.write('\n=====================================================================')
    sys.stdout.write('\nDriver Information:      ')
    sys.stdout.write('\n=====================================================================')
    sys.stdout.write('\nChosen CAE software:\n\t%s' % driver_name)
    if cae_software_version is not None:
        sys.stdout.write('\nVersion:\n\t%s' % cae_software_version)
    if docker_image is not None:
        sys.stdout.write(
            '\nAs requested, %s will be run in Docker containers based on'
            '\nthe following Docker image:\n\t%s.' % (driver_name, docker_image)
        )
    if post_post_file_prefix is not None:
        sys.stdout.write(
            '\nQuantities of interest will be extracted from result files indicated'
            '\nby the following (sub-)string:\n\t%s' % post_post_file_prefix
        )
    sys.stdout.write('\n=====================================================================')
    sys.stdout.write('\n')


def print_database_information(db, restart=False):
    """ Print out information on existing and newly established databases
    Args:
        restart (bool): Flag for the restart option of QUEENS

    """
    sys.stdout.write('\n=====================================================================')
    sys.stdout.write('\nDatabase Information:      ')
    sys.stdout.write('\n=====================================================================')
    sys.stdout.write('\nDatabase server:\n\t%s' % db.database_address)

    if db.drop_all_existing_dbs:
        sys.stdout.write('\nAs requested, all QUEENS databases for this user were dropped.')
    else:
        sys.stdout.write(
            '\nNumber of existing QUEENS databases for this user:\n\t%d' % len(db.database_list)
        )
        sys.stdout.write('\nList of existing QUEENS databases for this user:')
        for database in db.database_list:
            sys.stdout.write('\n\t%s' % database)

    if restart:
        if db.database_already_existent:
            sys.stdout.write('\nRestart:\n\tDatabase found.')
        else:
            sys.stdout.write('\nRestart:\n\tNo database found.')
    else:
        sys.stdout.write('\nEstablished new database:\n\t%s' % db.database_name)
        if db.database_already_existent:
            sys.stdout.write('\nCaution: note that the newly established database already existed!')

    sys.stdout.write('\n=====================================================================')
    sys.stdout.write('\n')
