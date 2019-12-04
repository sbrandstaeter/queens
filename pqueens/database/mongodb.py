
import pymongo
from pymongo.errors import ServerSelectionTimeoutError
from pqueens.utils.compression import compress_nested_container, decompress_nested_container

COMPRESS_TYPE = 'compressed array'

class MongoDB(object):
    """ MongoDB based database to store data of computer experiments

    Attributes:
        client (pymongo.MongoClient):   database client
        db (string):                    database
        myId (int):                     connection id
    """
    def __init__(self, database_address='localhost:27017',
                 database_name='pqueens', drop_existing_db=False):
        """
        Args:
            database_address (string): adress of database to connect to
            database_name (string):    name of database
            drop_existing_db (bool):   drop existing db if it exists
        """
        #try:
        self.client = pymongo.MongoClient(host=[database_address],
                                          serverSelectionTimeoutMS=10000)
        if drop_existing_db:
            self.client.drop_database(database_name)
        self.db = self.client[database_name]

        self.client.server_info() # Forces a call and raises
        # ServerSelectionTimeoutError exception:


    def save(self, save_doc, experiment_name, experiment_field, batch,
             field_filters=None):
        """ Save a document to the database.

        Any numpy arrays in the document are compressed so that they can be
        saved to MongoDB. field_filters must return at most one document,
        otherwise it is not clear which one to update and an exception will
        be raised.

        Args:
            save_doc (dict,list):       document to be saved to the db
            experiment_name (string):   experiment the data belongs to
            experiment_field (string):  experiment field data belongs to
            batch (string):             batch the data belongs to
            field_filters (dict):       filter to find appropriate document
                                        to create or update
        Returns:
            bool: is this the result of an acknowledged write operation ?
        """
        if field_filters is None:
            field_filters = {}

        save_doc = compress_nested_container(save_doc)

        dbcollection = self.db[experiment_name][batch][experiment_field]
        dbdocs = list(dbcollection.find(field_filters))

        upsert = False

        if len(dbdocs) > 1:
            raise Exception('Ambiguous save attempted. Field filters returned '
                            'more than one document.')
        elif len(dbdocs) == 1:
            dbdoc = dbdocs[0]
        else:
            upsert = True

        result = dbcollection.replace_one(field_filters, save_doc, upsert=upsert)

        return result.acknowledged

    def load(self, experiment_name, batch, experiment_field, field_filters=None):
        """ Load document(s) from the database

        Decompresses any numpy arrays

        Args:
            experiment_name (string):  experiment the data belongs to
            experiment_field (string): experiment field data belongs to
            batch (string):            batch the data belongs to
            field_filters (dict):      filter to find appropriate document(s)
                                       to load

        Returns:
            list: list of documents matching query
        """

        if field_filters is None:
            field_filters = {}

        dbcollection = self.db[experiment_name][batch][experiment_field]
        dbdocs = list(dbcollection.find(field_filters))

        if len(dbdocs) == 0:
            return None
        elif len(dbdocs) == 1:
            return decompress_nested_container(dbdocs[0])
        else:
            return [decompress_nested_container(dbdoc) for dbdoc in dbdocs]

    def remove(self, experiment_name, experiment_field, batch, field_filters={}):
        """ Remove a list of documents from the database

        Args:
            experiment_name (string):  experiment the data belongs to
            experiment_field (string): experiment field data belongs to
            batch (string):            batch the data belongs to
            field_filters (dict):      filter to find appropriate document(s)
                                       to delete
         """
        self.db[experiment_name][batch][experiment_field].delete_one(field_filters)
