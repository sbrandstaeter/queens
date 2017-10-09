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
        self.dummy_job['dummy_field1']='gargbage'
        self.dummy_job['dummy_field2']='rubbish'
        self.job_id =1

    def test_connection(self):
        # check if we can connect to mongodb
        #db = MongoDB(database_address="localhost:27017")
        db = MongoDB(database_address="mongodb:27017")
        self.assertIsInstance(db, MongoDB, 'First argument is not a MongoDB')

    def test_read_write_delete(self):
        db = MongoDB(database_address="localhost:27017")
        # save some dummy data
        db.save(self.dummy_job,self.experiment_name, 'jobs', {'id' : self.job_id})
        # try to retrieve it
        jobs  = db.load(self.experiment_name, 'jobs')
        if isinstance(jobs, dict):
            jobs = [jobs]

        test = jobs[0]['dummy_field1']
        self.assertEqual(test, 'gargbage')

        # remove dummy data
        db.remove( self.experiment_name,'jobs')
        jobs  = db.load(self.experiment_name, 'jobs')
        # assert that jobs is empty
        self.assertFalse(jobs)
