"""Module supplies functions for printing out information.

Is printed at the beginning of each QUEENS run.
"""

import logging

_logger = logging.getLogger(__name__)


def print_scheduling_information(scheduler_type, remote, remote_connect, singularity):
    """Print out information on chosen scheduling."""
    _logger.info('=================================================================')
    _logger.info('Scheduling Information:')
    _logger.info('=================================================================')
    _logger.info('Chosen type of scheduling:')
    _logger.info('\t%s', scheduler_type)
    if not remote:
        _logger.info('Jobs will be run on local computing resource.')
    else:
        _logger.info('Jobs will be run on remote computing resource as')
        _logger.info('\t%s', remote_connect)
    if singularity:
        _logger.info('All jobs will be run in Singularity containers.')
    else:
        if remote:
            _logger.info(
                'Please note that remote scheduling without using Singularity\n'
                'containers might result in enhanced network traffic!'
            )
    _logger.info('=================================================================')
    _logger.info('\n')


def print_driver_information(
    driver_type, cae_software_version, data_processor_file_prefix, docker_image
):
    """Print out information on chosen driver."""
    # determine name of driver
    driver_dict = {
        'mpi': 'MPI',
    }
    driver_name = driver_dict[driver_type]

    _logger.info('\n=====================================================================')
    _logger.info('\nDriver Information:      ')
    _logger.info('\n=====================================================================')
    _logger.info('\nChosen CAE software:\n\t%s', driver_name)
    if cae_software_version is not None:
        _logger.info('\nVersion:\n\t%s', cae_software_version)
    if docker_image is not None:
        _logger.info(
            '\nAs requested, %s will be run in Docker containers based on'
            '\nthe following Docker image:\n\t%s.',
            driver_name,
            docker_image,
        )
    if data_processor_file_prefix is not None:
        _logger.info(
            '\nQuantities of interest will be extracted from result files indicated'
            '\nby the following (sub-)string:\n\t%s',
            data_processor_file_prefix,
        )
    _logger.info('\n=====================================================================')
    _logger.info('\n')


def print_database_information(db, restart=False):
    """Print out information on existing and newly established databases.

    Args:
        db (obj): Database object
        restart (bool): Flag for the restart option of QUEENS
    """
    _logger.info('\n=====================================================================')
    _logger.info('\nDatabase Information:      ')
    _logger.info('\n=====================================================================')
    _logger.info('\nDatabase server:\n\t%s', db.database_address)

    if db.drop_all_existing_dbs:
        _logger.info('\nAs requested, all QUEENS databases for this user were dropped.')
    else:
        _logger.info(
            '\nNumber of existing QUEENS databases for this user:\n\t%d', len(db.database_list)
        )
        _logger.info('\nList of existing QUEENS databases for this user:')
        for database in db.database_list:
            _logger.info('\n\t%s', database)

    if restart:
        if db.database_already_existent:
            _logger.info('\nRestart:\n\tDatabase found.')
        else:
            _logger.info('\nRestart:\n\tNo database found.')
    else:
        _logger.info('\nEstablished new database:\n\t%s', db.database_name)
        if db.database_already_existent:
            _logger.info('\nCaution: note that the newly established database already existed!')

    _logger.info('\n=====================================================================')
    _logger.info('\n')
