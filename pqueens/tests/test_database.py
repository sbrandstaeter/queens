'''
Created on June 22nd 2017
@author: jbi

'''

import unittest
from pqueens.database.mongodb import MongoDB

class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.experiment_name = 'simple_test'
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

    def test_read_write_delete(self):
        try:
            db = MongoDB(database_address="localhost:27017")
        except:
            # if local host fails try to use alias if db is in docker container
            db = MongoDB(database_address="mongodb:27017")

        # save some dummy data
        db.save(self.dummy_job, self.experiment_name, 'jobs', {'id' : self.job_id})
        # try to retrieve it
        jobs = db.load(self.experiment_name, 'jobs')
        if isinstance(jobs, dict):
            jobs = [jobs]

        test = jobs[0]['dummy_field1']
        self.assertEqual(test, 'garbage')

        # remove dummy data
        db.remove(self.experiment_name, 'jobs')
        jobs = db.load(self.experiment_name, 'jobs')
        # assert that jobs is empty
        self.assertFalse(jobs)
