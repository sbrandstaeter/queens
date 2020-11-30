import sys
import getpass
import pymongo
import pandas as pd
from pymongo.errors import ServerSelectionTimeoutError
from pqueens.utils.compression import compress_nested_container, decompress_nested_container
import numpy as np
import xarray as xr

COMPRESS_TYPE = 'compressed array'


class MongoDB(object):
    """
    MongoDB based database to store data of computer experiments

    Attributes:
        db (obj): Mongo DB database object
        database_name (str): Name of the current database
        database_list (list): List with names of existing QUEENS databases
        database_already_existent (bool): Boolean which is True if database name already exists

    Returns:
        MongoDB_obj (obj): Instance of MongoDB class

    """

    def __init__(
        self,
        db,
        database_address,
        database_name,
        database_list,
        database_already_existent,
        drop_all_existing_dbs,
    ):
        self.db = db
        self.database_address = database_address
        self.database_name = database_name
        self.database_list = database_list
        self.database_already_existent = database_already_existent
        self.drop_all_existing_dbs = drop_all_existing_dbs

    @classmethod
    def from_config_create_database(cls, config):
        """
        Create Mongo database object from problem description

        Args:
            config (dict): Dictionary containing the problem description of the current QUEENS
                           simulation

        Returns:
            MongoDB_obj (obj): Instance of MongoDB class

        """
        try:
            database_name_final = config['global_settings'].get('experiment_name', 'dummy')
        # in case global settings do not exist
        except KeyError:
            database_name_final = 'dummy'

        database_address = config['database'].get('address', 'localhost:27017')
        drop_all_existing_dbs = config['database'].get('drop_all_existing_dbs', False)

        reset_database = config['database'].get('reset_database', False)

        client = pymongo.MongoClient(
            host=[database_address], serverSelectionTimeoutMS=100, connect=False
        )

        attempt = 0
        while attempt < 10:
            try:
                client.server_info()  # Forces a call
                break
            except pymongo.errors.ServerSelectionTimeoutError:
                if attempt == 9:
                    client.server_info()
                else:
                    print('ServerSelectionTimeoutError in mongodb.py')
            attempt += 1

        # get list of all existing databases
        complete_database_list = client.list_database_names()

        # generate name of database to be established for this QUEENS run
        user_name = getpass.getuser()
        database_name_prefix = 'queens_db_' + user_name
        database_name = database_name_prefix + '_' + database_name_final

        # declare boolean variable for existence of database to be established
        database_exists = False

        # declare list for QUEENS databases
        database_list = []

        # check all existent databases on whether they are QUEENS databases
        for check_database in complete_database_list:
            if database_name_prefix in check_database:
                # drop all existent QUEENS databases for this user if desired
                if drop_all_existing_dbs:
                    client.drop_database(check_database)
                else:
                    # add to list of existing QUEENS databases
                    database_list.append(check_database)
                    # check if database with this name already existed before
                    if check_database == database_name:
                        database_exists = True

        if database_exists and reset_database:
            client.drop_database(database_name)

        # establish new database for this QUEENS run
        db = client[database_name]

        return cls(
            db,
            database_address,
            database_name,
            database_list,
            database_exists,
            drop_all_existing_dbs,
        )

    def print_database_information(self, restart=False):
        """ Print out information onn existing and newly established databases
        Args:
            restart (bool): Flag for the restart option of QUEENS

        """
        sys.stdout.write('\n=====================================================================')
        sys.stdout.write('\nDatabase Information:      ')
        sys.stdout.write('\n=====================================================================')
        sys.stdout.write('\nDatabase server: %s' % self.database_address)

        if self.drop_all_existing_dbs:
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

    def save(self, save_doc, experiment_name, experiment_field, batch, field_filters={}):
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
        self._pack_labeled_data(save_doc)

        save_doc = compress_nested_container(save_doc)

        dbcollection = self.db[experiment_name][batch][experiment_field]

        # make sure that there will be only one document in the collection that fits field_filters
        # note that the empty field_filers={} fits all documents in a collection
        if dbcollection.count_documents(field_filters) > 1:
            raise Exception(
                'Ambiguous save attempted. Field filters returned more than one document.'
            )
        attempt = 0
        max_attempts = 10
        while attempt < max_attempts:
            attempt += 1
            try:
                result = dbcollection.replace_one(field_filters, save_doc, upsert=True)
                break
            except pymongo.errors.DuplicateKeyError as duplicate_key_error:
                print(
                    f'Caught exception pymongo.errors.DuplicateKeyError. '
                    f'Attempt: {attempt}/10.\n'
                )
                if attempt == max_attempts:
                    print("Reached maximum attempts. Raising error:\n")
                    raise duplicate_key_error
                else:
                    print('Retrying ...\n')
        return result.acknowledged

    def count_documents(self, experiment_name, batch, experiment_field, field_filters={}):
        """
        Return number of document(s) in collection

        Args:
            experiment_name (string):  experiment the data belongs to
            experiment_field (string): experiment field data belongs to
            batch (string):            batch the data belongs to
            field_filters (dict):      filter to find appropriate document(s)
                                       to load

        Returns:
            int: number of documents in collection
        """

        dbcollection = self.db[experiment_name][batch][experiment_field]
        doc_count = dbcollection.count_documents(field_filters)

        return doc_count

    def estimated_count(self, experiment_name, batch, experiment_field):
        """ Return estimated count of document(s) in collection

        This is very fast but not 100% safe.
        Args:
            experiment_name (string):  experiment the data belongs to
            experiment_field (string): experiment field data belongs to
            batch (string):            batch the data belongs to
            field_filters (dict):      filter to find appropriate document(s)
                                       to load

        Returns:
            int: number of documents in collection
        """

        dbcollection = self.db[experiment_name][batch][experiment_field]
        doc_count = dbcollection.estimated_document_count()

        return doc_count

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

        self._unpack_labeled_data(dbdocs)

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

    def _pack_labeled_data(self, save_doc):
        """
        Pack labeled data (e.g. pandas DataFrame or xarrays) for database

        Args:
              save_doc (dict): dictionary to be saved to database
        """
        if isinstance(save_doc.get('result', None), pd.DataFrame):
            self._pack_pandas_dataframe(save_doc)

        elif isinstance(save_doc.get('result', None), xr.DataArray) or isinstance(
            save_doc.get('result', None), xr.Dataset
        ):
            raise Exception('Packing method for xarrays not implemented.')

    @staticmethod
    def _pack_pandas_dataframe(save_doc):
        """
        Pack pandas DataFrame for database

        - reset index to recover the index in the unpacking method
        - convert DataFrame to mongodb compatible data type

        Args:
              save_doc (dict): dictionary to be saved to database
        """
        result = save_doc['result']

        number_of_index_levels = result.index.nlevels
        result.reset_index(inplace=True)
        column_header = np.array(result.columns)
        index_format = np.empty_like(column_header)
        index_format[0] = number_of_index_levels
        index_format[1] = 'pd.DataFrame'
        data = np.array(result)

        result = np.vstack((index_format, column_header, data))
        save_doc['result'] = result.tolist()

    def _unpack_labeled_data(self, dbdocs):
        """
        Unpack data from database to labeled data format (e.g. pandas DataFrame or xarrays)

        Args:
            dbdocs (list): documents from database
        """
        for idx in range(len(dbdocs)):
            current_doc = dbdocs[idx]
            current_result = current_doc.get('result', None)

            if isinstance(current_result, list) and (current_result[0][1] == 'pd.DataFrame'):
                data, index = self._split_output(current_result)
                current_doc['result'] = pd.DataFrame(data=data, index=index)

    @staticmethod
    def _split_output(result):
        """"
        Split output into (multi-)index and data

        Args:
            result (list): result as list with first row = header

        Returns:
            data (np.array): data as column vector
            index (pd.MultiIndex): multiindex containing coordinates
        """
        index_format = result[0][0]
        column_header = np.array(result[1])
        body = np.array(result[2:])

        coordinate_names = column_header[:index_format]

        index = None
        if index_format == 1:
            index = pd.Index(body[:, 0])
        elif index_format > 1:
            coordinates = list(zip(*body[:, :index_format].transpose()))
            index = pd.MultiIndex.from_tuples(coordinates, names=coordinate_names)
        else:
            Exception('No index found')

        data = body[:, index_format:]

        return data, index
