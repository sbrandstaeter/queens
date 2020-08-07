import sys
import getpass
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

    def __init__(
        self,
        database_name_final='dummy',
        database_address='localhost:27017',
        drop_existing_db=False,
    ):
        """
        Args:
            database_address (string): adress of database to connect to
            database_name (string):    name of database
            drop_existing_db (bool):   drop existing QUEENS databases of user if existent
        """
        # try:
        self.client = pymongo.MongoClient(host=[database_address], serverSelectionTimeoutMS=100)

        # get list of all existing databases
        complete_database_list = self.client.list_database_names()

        # generate name of database to be established for this QUEENS run
        user_name = getpass.getuser()
        database_name_prefix = 'queens_db_' + user_name
        self.database_name = database_name_prefix + '_' + database_name_final

        # declare boolean variable for existence of database to be established
        self.database_already_existent = False

        # declare list for QUEENS databases
        self.database_list = []

        # check all existent databases on whether they are QUEENS databases
        for check_database in complete_database_list:
            if database_name_prefix in check_database:
                # drop all existent QUEENS databases for this user if desired
                if drop_existing_db:
                    self.client.drop_database(check_database)
                else:
                    # add to list of existing QUEENS databases
                    self.database_list.append(check_database)
                    # check if database with this name already existed before
                    if check_database == self.database_name:
                        self.database_already_existent = True

        # establish new database for this QUEENS run
        self.db = self.client[self.database_name]

        attempt = 0
        while attempt < 10:
            try:
                self.client.server_info()  # Forces a call
                break
            except pymongo.errors.ServerSelectionTimeoutError:
                if attempt == 9:
                    self.client.server_info()
                else:
                    print('ServerSelectionTimeoutError in mongodb.py')
            attempt += 1

    def print_database_information(
        self, database_address='localhost:27017', drop_existing_db=False, restart=False,
    ):
        """ Print out information onn existing and newly established databases
        Args:
            drop_existing_db (bool):   drop existing QUEENS databases of user if existent

        """
        sys.stdout.write('\n=====================================================================')
        sys.stdout.write('\nDatabase Information:      ')
        sys.stdout.write('\n=====================================================================')
        sys.stdout.write('\nDatabase server: %s' % database_address)

        if drop_existing_db:
            sys.stdout.write('\nAs requested, all QUEENS databases for this user were dropped.')
        else:
            sys.stdout.write(
                '\nNumber of existing QUEENS databases for this user: %d' % len(self.database_list)
            )
            sys.stdout.write(
                '\nList of existing QUEENS databases for this user: %s' % self.database_list
            )

        if restart:
            if self.database_already_existent:
                sys.stdout.write('\nRestart: Database found.')
            else:
                sys.stdout.write('\nRestart: No database found.')
        else:
            sys.stdout.write('\nEstablished new database: %s' % self.database_name)
            if self.database_already_existent:
                sys.stdout.write(
                    '\nCaution: note that the newly established database already existed!'
                )

        sys.stdout.write('\n=====================================================================')
        sys.stdout.write('\n')

    def save(self, save_doc, experiment_name, experiment_field, batch, field_filters=None):
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
            raise Exception(
                'Ambiguous save attempted. Field filters returned more than one document.'
            )
        elif len(dbdocs) == 1:
            dbdoc = dbdocs[0]
        else:
            upsert = True

        attempt = 0
        while attempt < 10:
            try:
                result = dbcollection.replace_one(field_filters, save_doc, upsert=upsert)
                break
            except pymongo.errors.DuplicateKeyError:
                if attempt == 9:
                    result = dbcollection.replace_one(field_filters, save_doc, upsert=upsert)
                else:
                    print('Exception pymongo.errors.DuplicateKeyError occurred')
            attempt += 1

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
        self.db[experiment_name][batch][experiment_field].delete_many(field_filters)
