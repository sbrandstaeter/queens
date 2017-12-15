'''
Created on June 22nd 2017
@author: jbi

'''

import unittest
from pqueens.database.mongodb import MongoDB
from pymongo.errors import ServerSelectionTimeoutError

class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.experiment_name = 'simple_test'
        self.batch = 1
        self.dummy_job = {}
        self.dummy_job['dummy_field1'] = 'garbage'
        self.dummy_job['dummy_field2'] = 'rubbish'
        self.job_id = 1

    def test_connection(self):
        # check if we can connect to mongodb
        try:
            db = MongoDB(database_address="localhost:27017")
        except:
            # if local host fails try to use alias if db is in docker container
            db = MongoDB(database_address="mongodb:27017")

        self.assertIsInstance(db, MongoDB, 'First argument is not a MongoDB')

    def test_connection_fails(self):
        # check if we get correct exception when failing to connect mongodb
        with self.assertRaises(ServerSelectionTimeoutError):
            MongoDB(database_address="localhos:2016")


    def test_droppping(self):
        # check if we can connect to mongodb and clear preexisting db
        try:
            db = MongoDB(database_address="localhost:27017",drop_existing_db=True)
        except:
            # if local host fails try to use alias if db is in docker container
            db = MongoDB(database_address="mongodb:27017",drop_existing_db=True)

        self.assertIsInstance(db, MongoDB, 'First argument is not a MongoDB')


    def test_read_write_delete(self):
        try:
            db = MongoDB(database_address="localhost:27017",drop_existing_db=True)
        except:
            # if local host fails try to use alias if db is in docker container
            db = MongoDB(database_address="mongodb:27017",drop_existing_db=True)

        # save some dummy data
        db.save(self.dummy_job, self.experiment_name, 'jobs', self.batch, {'id' : self.job_id})
        db.save(self.dummy_job, self.experiment_name, 'jobs', self.batch)

        # try to retrieve it
        jobs = db.load(self.experiment_name, self.batch, 'jobs')
        if isinstance(jobs, dict):
            jobs = [jobs]

        test = jobs[0]['dummy_field1']
        self.assertEqual(test, 'garbage')

        # remove dummy data
        db.remove(self.experiment_name, 'jobs', self.batch)
        jobs = db.load(self.experiment_name, self.batch, 'jobs')
        # assert that jobs is empty
        self.assertFalse(jobs)

    def test_write_multiple_entries(self):
        try:
            db = MongoDB(database_address="localhost:27017",drop_existing_db=True)
        except:
            # if local host fails try to use alias if db is in docker container
            db = MongoDB(database_address="mongodb:27017",drop_existing_db=True)

        # save some dummy data
        db.save(self.dummy_job, self.experiment_name, 'jobs', self.batch)
        db.save(self.dummy_job, self.experiment_name, 'jobs', self.batch, {'id' : self.job_id})

        jobs = db.load(self.experiment_name, self.batch, 'jobs')
        if isinstance(jobs, dict):
            jobs = [jobs]
        self.assertEqual(len(jobs), 2)

        # should cause problems
        with self.assertRaises(Exception):
            db.save(self.dummy_job, self.experiment_name, 'jobs', self.batch)
