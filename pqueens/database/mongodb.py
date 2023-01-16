"""MongoDB module."""
import getpass
import logging

import numpy as np
import pandas as pd
import xarray as xr
from pymongo import MongoClient

from pqueens.database.database import Database
from pqueens.utils.decorators import safe_operation
from pqueens.utils.mongodb import (
    convert_nested_data_to_db_dict,
    convert_nested_db_dicts_to_lists_or_arrays,
)
from pqueens.utils.print_utils import get_str_table

_logger = logging.getLogger(__name__)
COMPRESS_TYPE = 'compressed array'


class MongoDB(Database):
    """MongoDB based database to store data of computer experiments.

    Attributes:
        db_name (str): Name of the current database
        reset_existing_db (boolean): Flag to reset the database if necessary
        db_obj (obj): Mongo DB database object
        db_address (str): Database address
        db_list (list): List with names of existing QUEENS databases
        db_already_existent (bool): Boolean which is True if database name already exists
        drop_all_existing_dbs (bool): Flag to drop all user databases if desired
        mongo_client (MongoClient): Mongodb client object
    """

    def __init__(
        self,
        db_name,
        reset_existing_db,
        db_obj=None,
        db_address="localhost:27017",
        db_list=None,
        db_already_existent=False,
        drop_all_existing_dbs=False,
        mongo_client=None,
    ):
        """Initialize mongodb object.

        Args:
            db_name (str): Name of the current database
            reset_existing_db (boolean): Flag to reset the database if necessary
            db_obj (obj): Mongo DB database object
            db_address (str): Database address
            db_list (list): List with names of existing QUEENS databases
            db_already_existent (bool): Boolean which is True if database name already exists
            drop_all_existing_dbs (bool): Flag to drop all user databases if desired
            mongo_client (MongoClient): Mongodb client object
        Returns:
            MongoDB (obj): Instance of MongoDB class
        """
        super().__init__(db_name, reset_existing_db)
        self.db_address = db_address
        self.db_obj = db_obj
        self.db_list = db_list
        self.db_already_existent = db_already_existent
        self.drop_all_existing_dbs = drop_all_existing_dbs
        self.mongo_client = mongo_client

    @classmethod
    def from_config_create_database(cls, config):
        """Create Mongo database object from problem description.

        Args:
            config (dict): Dictionary containing the problem description of the current QUEENS
                           simulation

        Returns:
            MongoDB (obj): Instance of MongoDB class
        """
        db_name = config['database'].get('name')

        #  if the database name is not defined in the input file, create a unique name now
        # TODO fix this
        if not db_name:
            try:
                db_name_suffix = config['global_settings'].get('experiment_name', 'dummy')
            # in case global settings do not exist
            except KeyError:
                db_name_suffix = 'dummy'
                _logger.warning("Global settings missing, db_suffix was set to 'dummy'")

            user_name = getpass.getuser()
            db_name_prefix = 'queens_db_' + user_name
            db_name = db_name_prefix + '_' + db_name_suffix

        reset_existing_db = config['database'].get('reset_existing_db', True)
        drop_all_existing_dbs = config['database'].get('drop_all_existing_dbs', False)
        db_address = config['database'].get('address', 'localhost:27017')
        db_name_prefix = db_name

        # Pass empty arguments. These are later established in the _connect function
        mongo_client = None
        db_obj = None
        db_list = None
        db_already_existent = False

        return cls(
            db_name=db_name,
            reset_existing_db=reset_existing_db,
            db_obj=db_obj,
            db_address=db_address,
            db_list=db_list,
            db_already_existent=db_already_existent,
            drop_all_existing_dbs=drop_all_existing_dbs,
            mongo_client=mongo_client,
        )

    @safe_operation
    def _connect(self):
        """Connect to the database."""
        # Construct the Mongodb client
        self.mongo_client = MongoClient(
            host=[self.db_address], serverSelectionTimeoutMS=1000, connect=False
        )

        # Try the connection
        self.mongo_client.server_info()

        self.db_obj = self.mongo_client[self.db_name]

        _logger.info("Connected to %s", self.db_address)

    def _disconnect(self):
        """Mongodb automatically closes the connection."""
        _logger.info("Disconnected the database")

    def _clean_database(self):
        """If desired reset the current database."""
        # get list of all existing databases
        complete_db_list = self.mongo_client.list_database_names()

        # Check if db exists already
        db_exists = self.db_name in complete_db_list
        self.db_already_existent = db_exists

        # Reset the database if desired
        if db_exists and self.reset_existing_db:
            self._delete_database(self.db_name)

        self.db_list = complete_db_list

    def _delete_database(self, db_name):
        """Remove database from the database server.

        Args:
            db_name (str): database name to be deleted
        """
        self.mongo_client.drop_database(db_name)
        _logger.info("%s was dropped", db_name)

    def _delete_databases_by_prefix(self, prefix):
        """Remove databases from the database server by prefix.

        Args:
            prefix (str): Databases with this prefix in the name are deleted
        """
        # get list of all existing databases
        complete_db_list = self.mongo_client.list_database_names()

        # List of dbs with this prefix
        db_to_deleted_list = [db for db in complete_db_list if not db.find(prefix)]

        for db in db_to_deleted_list:
            self.mongo_client.drop_database(db)

        _logger.info("Databases with prefix %s were deleted!", prefix)

    @safe_operation
    def save(self, save_doc, experiment_name, experiment_field, batch, field_filters=None):
        """Save a document to the database.

        Any numpy arrays in the document are compressed so that they can be
        saved to MongoDB. field_filters must return at most one document,
        otherwise it is not clear which one to update and an exception will
        be raised.

        Args:
            save_doc (dict,list):       document to be saved to the db
            experiment_name (string):   experiment the data belongs to
            experiment_field (string):  experiment field data belongs to
            batch (int):                batch the data belongs to
            field_filters (dict):       filter to find appropriate document
                                        to create or update
        Returns:
            bool: is this the result of an acknowledged write operation ?
        """
        if field_filters is None:
            field_filters = {}
        self._pack_labeled_data(save_doc)

        save_doc = convert_nested_data_to_db_dict(save_doc)

        dbcollection = self.db_obj[experiment_name][batch][experiment_field]

        # make sure that there will be only one document in the collection that fits field_filters
        # note that the empty field_filers={} fits all documents in a collection
        if dbcollection.count_documents(field_filters) > 1:
            raise Exception(
                'Ambiguous save attempted. Field filters returned more than one document.'
            )
        result = dbcollection.replace_one(field_filters, save_doc, upsert=True)
        return result.acknowledged

    @safe_operation
    def count_documents(self, experiment_name, batch, experiment_field, field_filters=None):
        """Return number of document(s) in collection.

        Args:
            experiment_name (string):  experiment the data belongs to
            experiment_field (string): experiment field data belongs to
            batch (int):               batch the data belongs to
            field_filters (dict):      filter to find appropriate document(s)
                                       to load

        Returns:
            int: number of documents in collection
        """
        if field_filters is None:
            field_filters = {}

        dbcollection = self.db_obj[experiment_name][batch][experiment_field]
        doc_count = dbcollection.count_documents(field_filters)

        return doc_count

    @safe_operation
    def estimated_count(self, experiment_name, batch, experiment_field):
        """Return estimated count of document(s) in collection.

        This is very fast but not 100% safe.
        Args:
            experiment_name (string):  experiment the data belongs to
            experiment_field (string): experiment field data belongs to
            batch (int):               batch the data belongs to
            field_filters (dict):      filter to find appropriate document(s)
                                       to load

        Returns:
            int: number of documents in collection
        """
        dbcollection = self.db_obj[experiment_name][batch][experiment_field]
        doc_count = dbcollection.estimated_document_count()

        return doc_count

    @safe_operation
    def load(self, experiment_name, batch, experiment_field, field_filters=None):
        """Load document(s) from the database.

        Decompresses any numpy arrays

        Args:
            experiment_name (string):  experiment the data belongs to
            experiment_field (string): experiment field data belongs to
            batch (int):               batch the data belongs to
            field_filters (dict):      filter to find appropriate document(s)
                                       to load

        Returns:
            list: list of documents matching query
        """
        if field_filters is None:
            field_filters = {}

        dbcollection = self.db_obj[experiment_name][batch][experiment_field]
        dbdocs = list(dbcollection.find(field_filters))

        self._unpack_labeled_data(dbdocs)

        if len(dbdocs) == 0:
            return None
        elif len(dbdocs) == 1:
            return convert_nested_db_dicts_to_lists_or_arrays(dbdocs[0])
        else:
            return [convert_nested_db_dicts_to_lists_or_arrays(dbdoc) for dbdoc in dbdocs]

    @safe_operation
    def remove(self, experiment_name, experiment_field, batch, field_filters=None):
        """Remove a list of documents from the database.

        Args:
            experiment_name (string):  experiment the data belongs to
            experiment_field (string): experiment field data belongs to
            batch (int):               batch the data belongs to
            field_filters (dict):      filter to find appropriate document(s)
                                       to delete
        """
        if field_filters is None:
            field_filters = {}
        self.db_obj[experiment_name][batch][experiment_field].delete_many(field_filters)

    def _pack_labeled_data(self, save_doc):
        """Pack labeled data (e.g. pandas DataFrame or xarrays) for database.

        Args:
              save_doc (dict): dictionary to be saved to database
        """
        if isinstance(save_doc.get('result', None), pd.DataFrame):
            self._pack_pandas_dataframe(save_doc)

        elif isinstance(save_doc.get('result', None), (xr.DataArray, xr.Dataset)):
            raise Exception('Packing method for xarrays not implemented.')

    @staticmethod
    def _pack_pandas_dataframe(save_doc):
        """Pack pandas DataFrame for database.

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
        """Unpack data from database to labeled data format.

        Args:
            dbdocs (list): documents from database
        """
        # TODO: we should use a return statement, reference based is a dangerous game
        for idx in range(len(dbdocs)):
            current_doc = dbdocs[idx]
            current_result = current_doc.get('result', None)

            if isinstance(current_result, list) and isinstance(current_result[0], list):
                if current_result[0][1] == 'pd.DataFrame':
                    data, index = self._split_output(current_result)
                    current_doc['result'] = pd.DataFrame(data=data, index=index)

    @staticmethod
    def _split_output(result):
        """Split output into (multi-)index and data.

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

    def __str__(self):
        """Creates a string describing the MongoDB object.

        Returns:
            string (str): MongoDB obj description
        """
        name = "QUEENS MongoDB object wrapper"
        print_dict = {"Database address": self.db_address, "Database name ": self.db_name}
        return get_str_table(name, print_dict)
