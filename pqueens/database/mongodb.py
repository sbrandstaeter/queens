
import pymongo
from .abstractdb               import AbstractDB
from pqueens.utils.compression import compress_nested_container, decompress_nested_container

COMPRESS_TYPE = 'compressed array'

class MongoDB(AbstractDB):
    """ MongoDB based database to store data of computer experiments

    Attributes:
        client (pymongo.MongoClient):   database client
        db (string):                    database
        myId (int):                     connection id
    """
    def __init__(self, database_address='localhost:27017', database_name='pqueens'):
        """
        Args:
            database_address (string): adress of database to connect to
            database_name (string):    name of database
        """
        try:
            self.client = pymongo.MongoClient(host=[database_address])
            self.db     = self.client[database_name]

            # Get the ID of this connection for locking.
            self.myId = self.db.last_status()['connectionId']
        except:
            raise Exception('Could not connect to MongoDB.')

    def save(self, save_doc, experiment_name, experiment_field,
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
            field_filters (dict):       filter to find appropriate document
                                        to create or update
        """
        if field_filters is None:
            field_filters = {}

        save_doc = compress_nested_container(save_doc)

        dbcollection = self.db[experiment_name][experiment_field]
        dbdocs       = list(dbcollection.find(field_filters))

        upsert = False

        if len(dbdocs) > 1:
            raise Exception('Ambiguous save attempted. Field filters returned '
                            'more than one document.')
        elif len(dbdocs) == 1:
            dbdoc = dbdocs[0]
        else:
            #sys.stderr.write('Document not found, inserting new document.\n')
            upsert = True

        result = dbcollection.update(field_filters, save_doc, upsert=upsert)

        if upsert:
            return result['upserted']
        else:
            return result['updatedExisting']

    def load(self, experiment_name, experiment_field, field_filters=None):
        """ Load document(s) from the database

        Decompresses any numpy arrays

        Args:
            experiment_name (string):  experiment the data belongs to
            experiment_field (string): experiment field data belongs to
            field_filters (dict):      filter to find appropriate document(s)
                                       to load

        Returns:
            list: list of documents matching query
        """

        if field_filters is None:
            field_filters = {}

        dbcollection = self.db[experiment_name][experiment_field]
        dbdocs       = list(dbcollection.find(field_filters))

        if len(dbdocs) == 0:
            return None
        elif len(dbdocs) == 1:
            return decompress_nested_container(dbdocs[0])
        else:
            return [decompress_nested_container(dbdoc) for dbdoc in dbdocs]

    def remove(self, experiment_name, experiment_field, field_filters={}):
        """ Remove a list of documents from the database

        Args:
            experiment_name (string):  experiment the data belongs to
            experiment_field (string): experiment field data belongs to
            field_filters (dict):      filter to find appropriate document(s)
                                       to delete
         """
        self.db[experiment_name][experiment_field].remove(field_filters)
