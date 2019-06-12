'''
Created on June 22nd 2017
@author: jbi

'''
from pymongo.errors import ServerSelectionTimeoutError
import pytest

from pqueens.database.mongodb import MongoDB


@pytest.fixture(scope='module')
def dummy_job():
    """ A dummy job for the database. """
    dummy_job = {}
    dummy_job['dummy_field1'] = 'garbage'
    dummy_job['dummy_field2'] = 'rubbish'

    return dummy_job

@pytest.fixture(scope='module')
def experiment_name():
    return 'simple_test'


@pytest.fixture(scope='module')
def job_id():
    return 1


@pytest.fixture(scope='module')
def batch_id_1():
    return 1


@pytest.fixture(scope='module')
def batch_id_2():
    return 2


def test_connection():
    # check if we can connect to mongodb
    try:
        db = MongoDB(database_address="localhost:27017")
    except:
        # if local host fails try to use alias if db is in docker container
        db = MongoDB(database_address="mongodb:27017")

    assert isinstance(db, MongoDB)


def test_connection_fails():
    # check if we get correct exception when failing to connect mongodb
    with pytest.raises(ServerSelectionTimeoutError):
        MongoDB(database_address="localhos:2016")


def test_dropping():
    # check if we can connect to mongodb and clear preexisting db
    try:
        db = MongoDB(database_address="localhost:27017",drop_existing_db=True)
    except:
        # if local host fails try to use alias if db is in docker container
        db = MongoDB(database_address="mongodb:27017",drop_existing_db=True)

    assert isinstance(db, MongoDB)


def test_read_write_delete(dummy_job, experiment_name, batch_id_1, job_id):
    try:
        db = MongoDB(database_address="localhost:27017",drop_existing_db=True)
    except:
        # if local host fails try to use alias if db is in docker container
        db = MongoDB(database_address="mongodb:27017",drop_existing_db=True)

    # save some dummy data
    db.save(dummy_job, experiment_name, 'jobs', batch_id_1, {'id' : job_id})
    db.save(dummy_job, experiment_name, 'jobs', batch_id_1)

    # try to retrieve it
    jobs = db.load(experiment_name, batch_id_1, 'jobs')
    if isinstance(jobs, dict):
        jobs = [jobs]

    test = jobs[0]['dummy_field1']
    assert test ==  'garbage'

    # remove dummy data
    db.remove(experiment_name, 'jobs', batch_id_1)
    jobs = db.load(experiment_name, batch_id_1, 'jobs')
    # assert that jobs is empty
    assert not jobs


def test_write_multiple_entries(dummy_job, experiment_name, batch_id_2, job_id):

    try:
        db = MongoDB(database_address="localhost:27017",drop_existing_db=True)
    except:
        # if local host fails try to use alias if db is in docker container
        db = MongoDB(database_address="mongodb:27017",drop_existing_db=True)

    # save some dummy data
    db.save(dummy_job, experiment_name, 'jobs', batch_id_2)
    db.save(dummy_job, experiment_name, 'jobs', batch_id_2, {'id' : job_id})

    jobs = db.load(experiment_name, batch_id_2, 'jobs')
    if isinstance(jobs, dict):
        jobs = [jobs]
    assert len(jobs) is 2

    # should cause problems
    with pytest.raises(Exception):
        db.save(dummy_job, experiment_name, 'jobs', batch_id_2)
