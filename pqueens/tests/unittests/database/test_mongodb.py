'''
Created on June 22nd 2017
@author: jbi

'''
from pymongo.errors import ServerSelectionTimeoutError
import pytest
import pandas as pd
import numpy as np
import xarray as xr

from pqueens.database.mongodb import MongoDB


@pytest.fixture(scope='module')
def dummy_job():
    """ A dummy job for the database. """
    dummy_job = {}
    dummy_job['dummy_field1'] = 'garbage'
    dummy_job['dummy_field2'] = 'rubbish'

    return dummy_job


@pytest.fixture(scope='module')
def dummy_output_simple_index():
    output_simple_index = [
        [1, 'pd.DataFrame', None],
        ['my_index', 0, 1],
        [0.00000000e00, 3.50000000e-01, 4.85410116e-05],
        [5.00000000e-02, 3.50000000e-01, 5.11766167e-05],
    ]

    return output_simple_index


@pytest.fixture(scope='module')
def dummy_output_multi_index():
    output_multi_index = [
        [2, 'pd.DataFrame', None],
        ['name0', 'name1', 0],
        [0.00000000e00, 3.50000000e-01, 4.85410116e-05],
        [5.00000000e-02, 3.50000000e-01, 5.11766167e-05],
    ]

    return output_multi_index


@pytest.fixture(scope='module')
def dummy_job_with_result(dummy_output_multi_index):
    """ A dummy job for the database. """
    dummy_job_with_result = {0: {}, 1: {}}
    dummy_job_with_result[0]['dummy_field1'] = 'garbage'
    dummy_job_with_result[0]['dummy_field2'] = 'rubbish'
    dummy_job_with_result[0]['result'] = dummy_output_multi_index
    dummy_job_with_result[1]['dummy_field1'] = 'garbage'
    dummy_job_with_result[1]['dummy_field2'] = 'rubbish'
    dummy_job_with_result[1]['result'] = dummy_output_multi_index

    return dummy_job_with_result


@pytest.fixture(scope='module')
def dummy_job_with_list(dummy_output_multi_index):
    """ A dummy job with real list as result for the database. """
    dummy_job_with_list = {0: {}}
    dummy_job_with_list[0]['dummy_field1'] = 'garbage'
    dummy_job_with_list[0]['dummy_field2'] = 'rubbish'
    dummy_job_with_list[0]['result'] = [[0, 1, 2],
                                        [3, 4, 5]]

    return dummy_job_with_list


@pytest.fixture(scope='module')
def data_pandas_simple_index():
    data_simple_index = pd.DataFrame(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), index=['a', 'b', 'c']
    )
    return data_simple_index


@pytest.fixture(scope='module')
def data_pandas_multi_index():
    arrays = [
        ['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
        ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two'],
    ]
    tuples = list(zip(*arrays))
    index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
    np.random.seed(1)
    data_multi_index = pd.DataFrame(np.random.randn(8), index=index)

    return data_multi_index


@pytest.fixture(scope='module')
def data_xarray_dataarray():
    data = np.random.rand(4, 3)
    locs = ["IA", "IL", "IN"]
    times = pd.date_range("2000-01-01", periods=4)
    data_xarray_dataarray = xr.DataArray(data, coords=[times, locs], dims=["time", "space"])

    return data_xarray_dataarray


@pytest.fixture(scope='module')
def dummy_doc_with_pandas_multi(data_pandas_multi_index):
    """ A dummy doc for the database. """
    dummy_doc_with_pandas_multi = {
        'dummy_field1': 'garbage',
        'dummy_field2': 'rubbish',
        'result': data_pandas_multi_index,
    }

    return dummy_doc_with_pandas_multi


@pytest.fixture(scope='module')
def dummy_doc_with_pandas_simple(data_pandas_simple_index):
    """ A dummy doc for the database. """
    dummy_doc_with_pandas_simple = {
        'dummy_field1': 'garbage',
        'dummy_field2': 'rubbish',
        'result': data_pandas_simple_index,
    }

    return dummy_doc_with_pandas_simple


@pytest.fixture(scope='module')
def dummy_doc_with_xarray_dataarray(data_xarray_dataarray):
    """ A dummy doc for the database. """
    dummy_doc_with_xarray_dataarray = {
        'dummy_field1': 'garbage',
        'dummy_field2': 'rubbish',
        'result': data_xarray_dataarray,
    }

    return dummy_doc_with_xarray_dataarray


@pytest.fixture(scope='module')
def experiment_name():
    return 'mongodb_unittests'


@pytest.fixture(scope='module')
def job_id():
    return 1


@pytest.fixture(scope='module')
def batch_id_1():
    return 1


@pytest.fixture(scope='module')
def batch_id_2():
    return 2


@pytest.fixture(scope="session")
def username(session_mocker):
    """
    Make sure username returned by getuser is a dummy test username.

    This is necessary to ensure that the database tests do not interfere with the actual databases
    of the user calling the tests.
    """
    username = "pytest"
    session_mocker.patch('getpass.getuser', return_value=username)


def test_connection(username):
    """
    Test connection to mongodb service.

    Args:
        username: mock username for safety reasons

    """

    try:
        db = MongoDB.from_config_create_database({"database": {"address": "localhost:27017"}})
    except:
        # if local host fails try to use alias if db is in docker container
        db = MongoDB.from_config_create_database({"database": {"address": "mongodb:27017"}})

    assert isinstance(db, MongoDB)


def test_connection_fails(username):
    """
    Test for correct exception in case of failing connection to MongoDB service.

    Args:
        username: mock username for safety reasons

    Returns:

    """
    with pytest.raises(ServerSelectionTimeoutError):
        MongoDB.from_config_create_database({"database": {"address": "localhos:2016"}})


def test_read_write_delete(username, dummy_job, experiment_name, batch_id_1, job_id):
    """
    Test reading and writing to the database

    Args:
        username: mock username for safety reasons
        dummy_job: a test job that will be written and read to/from the database
        experiment_name: mock the experiment name of a QUEENS run needed for database name
        batch_id_1: (int) batch id needed for database name
        job_id: (int) id of the dummy_job needed for database name and database interaction

    Returns:

    """
    try:
        db = MongoDB.from_config_create_database(
            {
                "global_settings": {"experiment_name": experiment_name},
                "database": {"address": "localhost:27017", "drop_existing": True},
            }
        )
    except:
        # if local host fails try to use alias if db is in docker container
        db = MongoDB.from_config_create_database(
            {
                "global_settings": {"experiment_name": experiment_name},
                "database": {"address": "mongodb:27017", "drop_existing": True},
            }
        )

    # save some dummy data
    # save non-existing job with field_filter that will not match any entry -> insert new document
    db.save(dummy_job, experiment_name, 'jobs', batch_id_1, {'id': job_id})
    # empty field_filter will result in the insertion of a new document
    # after this there will be two documents in the collection
    db.save(dummy_job, experiment_name, 'jobs', batch_id_1)

    # try to retrieve it
    jobs = db.load(experiment_name, batch_id_1, 'jobs')
    if isinstance(jobs, dict):
        jobs = [jobs]

    test = jobs[0]['dummy_field1']
    assert test == 'garbage'

    # remove dummy data
    db.remove(experiment_name, 'jobs', batch_id_1)
    jobs = db.load(experiment_name, batch_id_1, 'jobs')
    # assert that jobs is empty
    assert not jobs


def test_write_multiple_entries(username, dummy_job, experiment_name, batch_id_2, job_id):

    try:
        db = MongoDB.from_config_create_database(
            {"database": {"address": "localhost:27017", "drop_existing": True}}
        )
    except:
        # if local host fails try to use alias if db is in docker container
        db = MongoDB.from_config_create_database(
            {"database": {"address": "mongodb:27017", "drop_existing": True}}
        )

    # save some dummy data
    db.save(dummy_job, experiment_name, 'jobs', batch_id_2)
    db.save(dummy_job, experiment_name, 'jobs', batch_id_2, {'id': job_id})

    jobs = db.load(experiment_name, batch_id_2, 'jobs')
    if isinstance(jobs, dict):
        jobs = [jobs]
    assert len(jobs) == 2

    # should cause problems
    with pytest.raises(Exception):
        db.save(dummy_job, experiment_name, 'jobs', batch_id_2, {"dummy_field1": "garbage"})


def test_pack_pandas_multi_index(dummy_doc_with_pandas_multi):
    try:
        db = MongoDB.from_config_create_database(
            {"database": {"address": "localhost:27017", "drop_existing": True}}
        )
    except ServerSelectionTimeoutError:
        # if local host fails try to use alias if db is in docker container
        db = MongoDB.from_config_create_database(
            {"database": {"address": "mongodb:27017", "drop_existing": True}}
        )

    db._pack_pandas_dataframe(dummy_doc_with_pandas_multi)
    db.save(dummy_doc_with_pandas_multi, 'dummy', 'jobs', 1)
    assert isinstance(dummy_doc_with_pandas_multi['result'], list)

    expected_format = np.array(
        [
            [2, 'pd.DataFrame', None],
            ['first', 'second', 0],
            ['bar', 'one', 1.6243453636632417],
            ['bar', 'two', -0.6117564136500754],
        ]
    )
    np.testing.assert_array_equal(
        np.array(dummy_doc_with_pandas_multi['result'][:4]), expected_format
    )


def test_pack_pandas_simple_index(dummy_doc_with_pandas_simple):
    try:
        db = MongoDB.from_config_create_database(
            {"database": {"address": "localhost:27017", "drop_existing": True}}
        )
    except ServerSelectionTimeoutError:
        # if local host fails try to use alias if db is in docker container
        db = MongoDB.from_config_create_database(
            {"database": {"address": "mongodb:27017", "drop_existing": True}}
        )

    db._pack_pandas_dataframe(dummy_doc_with_pandas_simple)
    db.save(dummy_doc_with_pandas_simple, 'dummy', 'jobs', 1)
    assert isinstance(dummy_doc_with_pandas_simple['result'], list)
    expected_format = np.array(
        [
            [1, 'pd.DataFrame', None, None],
            ['index', 0, 1, 2],
            ['a', 1, 2, 3],
            ['b', 4, 5, 6],
        ]
    )
    np.testing.assert_array_equal(
        np.array(dummy_doc_with_pandas_simple['result'][:4]), expected_format
    )


def test_pack_xarrays(dummy_doc_with_xarray_dataarray):
    try:
        db = MongoDB.from_config_create_database(
            {"database": {"address": "localhost:27017", "drop_existing": True}}
        )
    except ServerSelectionTimeoutError:
        # if local host fails try to use alias if db is in docker container
        db = MongoDB.from_config_create_database(
            {"database": {"address": "mongodb:27017", "drop_existing": True}}
        )

    # should cause problems: missing packing method for xarrays
    with pytest.raises(Exception):
        db._pack_labeled_data(dummy_doc_with_xarray_dataarray, 'dummy', 'jobs', 1)


def test_unpack_labeled_data(dummy_job_with_result):
    try:
        db = MongoDB.from_config_create_database(
            {"database": {"address": "localhost:27017", "drop_existing": True}}
        )
    except ServerSelectionTimeoutError:
        # if local host fails try to use alias if db is in docker container
        db = MongoDB.from_config_create_database(
            {"database": {"address": "mongodb:27017", "drop_existing": True}}
        )
    db._unpack_labeled_data(dummy_job_with_result)

    assert isinstance(dummy_job_with_result[0]['result'], pd.DataFrame)
    assert isinstance(dummy_job_with_result[1]['result'], pd.DataFrame)


def test_unpack_list(dummy_job_with_list):
    try:
        db = MongoDB.from_config_create_database(
            {"database": {"address": "localhost:27017", "drop_existing": True}}
        )
    except ServerSelectionTimeoutError:
        # if local host fails try to use alias if db is in docker container
        db = MongoDB.from_config_create_database(
            {"database": {"address": "mongodb:27017", "drop_existing": True}}
        )
    db._unpack_labeled_data(dummy_job_with_list)

    assert isinstance(dummy_job_with_list[0]['result'], list)


def test_split_output_no_index():

    output_no_index = [
        [None, None, None],
        ['my_index', 0, 1],
        [0.00000000e00, 3.50000000e-01, 4.85410116e-05],
        [5.00000000e-02, 3.50000000e-01, 5.11766167e-05],
    ]

    # should cause problems: missing packing method for xarrays
    with pytest.raises(Exception):
        MongoDB._split_output(output_no_index)


def test_split_output_simple_index(dummy_output_simple_index):

    data, index = MongoDB._split_output(dummy_output_simple_index)
    expected_data = np.array([[3.50000000e-01, 4.85410116e-05], [3.50000000e-01, 5.11766167e-05]])
    np.testing.assert_array_equal(data, expected_data)


def test_split_output_multi_index(dummy_output_multi_index):

    data, index = MongoDB._split_output(dummy_output_multi_index)
    expected_data = np.array([4.85410116e-05, 5.11766167e-05]).flatten()
    np.testing.assert_array_equal(data.flatten(), expected_data)
