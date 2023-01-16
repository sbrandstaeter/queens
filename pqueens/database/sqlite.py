"""Sqlite module."""
import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from pqueens.database.database import Database, QUEENSDatabaseError
from pqueens.utils.dictionary_utils import get_value_in_nested_dictionary
from pqueens.utils.print_utils import get_str_table
from pqueens.utils.sqlite import (
    boolean_from_binary,
    boolean_to_binary,
    np_array_from_binary,
    np_array_to_binary,
    obj_from_binary,
    obj_to_binary,
    pd_dataframe_from_binary,
    pd_dataframe_to_binary,
    safe_sqlitedb_operation,
    type_to_sqlite,
)

_logger = logging.getLogger(__name__)


# Add the adapters for different types to sqlite
sqlite3.register_adapter(np.ndarray, np_array_to_binary)
sqlite3.register_adapter(xr.DataArray, obj_to_binary)
sqlite3.register_adapter(pd.DataFrame, pd_dataframe_to_binary)
sqlite3.register_adapter(list, obj_to_binary)
sqlite3.register_adapter(dict, obj_to_binary)
sqlite3.register_adapter(bool, boolean_to_binary)

# Add the converters, i.e. back to the objects
sqlite3.register_converter("NPARRAY", np_array_from_binary)
sqlite3.register_converter("XARRAY", obj_from_binary)
sqlite3.register_converter("PDDATAFRAME", pd_dataframe_from_binary)
sqlite3.register_converter("LIST", obj_from_binary)
sqlite3.register_converter("DICT", obj_from_binary)
sqlite3.register_converter("BOOLEAN", boolean_from_binary)


class SQLite(Database):
    """SQLite wrapper for QUEENS.

    Attributes:
        db_name (str): Database name
        reset_existing_db (bool): Bool to reset database if desired
        database_path (Pathlib.Path): Path to database object
        tables (dict): dict of tables containing a dict with their column names and types
    """

    @classmethod
    def from_config_create_database(cls, config):
        """From config create database.

        Args:
            config (dict): Problem description

        Returns:
            sqlite database object
        """
        db_name = config['database'].get('name')
        if not db_name:
            db_name = config['global_settings'].get('experiment_name', 'dummy')

        db_path = config['database'].get('file')
        if db_path is None:
            db_path = Path(config["global_settings"]["output_dir"]).joinpath(db_name + ".sqlite.db")
            _logger.info(
                "No path for the sqlite database was provided, defaulting to %s", db_path.resolve()
            )
        else:
            db_path = Path(db_path)
        reset_existing_db = config['database'].get('reset_existing_db', True)

        # Check if the QUEENS run is remote
        if get_value_in_nested_dictionary(config, "remote", False):
            raise NotImplementedError(
                "QUEENS with Sqlite can currently not be used for remote computations! Switch"
                " to MongoDB if available"
            )

        return cls(db_name=db_name, reset_existing_db=reset_existing_db, database_path=db_path)

    def __init__(self, db_name, reset_existing_db, database_path):
        """Initialise database.

        Args:
            db_name (str): Database name
            reset_existing_db (bool): Bool to reset database if desired
            database_path (Pathlib.Path): Path to database object
        """
        super().__init__(db_name, reset_existing_db)
        self.database_path = database_path
        self.tables = {}

    def _connect(self):
        """Connect to the database.

        There is no need connect, so we just check the the connection
        here.
        """
        try:
            connection = sqlite3.connect(
                self.database_path, detect_types=sqlite3.PARSE_DECLTYPES, check_same_thread=False
            )
            connection.cursor()
            _logger.info("Connected to %s", self.database_path)
        except Exception as exception:
            raise QUEENSDatabaseError(
                "Could not connect to sqlite database from path {self.database_path}"
            ) from exception

    @safe_sqlitedb_operation
    def _execute(self, query, commit=False, parameters=None):
        """Execute query.

        Args:
            query (str): Query to be executed
            commit (bool, optional): Commit to the connection. Defaults to False.
            parameters (tuple, optional): Parameters to inject into the query. Defaults to None.

        Returns:
            sqlite.cursor: database cursor
        """
        connection = sqlite3.connect(
            self.database_path, detect_types=sqlite3.PARSE_DECLTYPES, check_same_thread=False
        )
        cursor = connection.cursor()
        if parameters:
            cursor.execute(query, parameters)
        else:
            cursor.execute(query)
        if commit:
            connection.commit()
        return cursor

    def _disconnect(self):
        """Disconnect the database."""
        _logger.info("Disconnected the database")

    def _load_table_names(self):
        """Get all table names in the database file.

        Returns:
            list: Name of tables
        """
        query = "SELECT name FROM sqlite_schema WHERE type='table';"
        cursor = self._execute(query)
        return [table_name[0] for table_name in cursor.fetchall()]

    def _delete_table(self, table_name):
        """Delete table by name.

        Args:
            table_name (table_name): Name of the table
        """
        query = f"DROP TABLE {table_name};"
        self._execute(query, commit=True)

        if table_name in self.tables.keys():
            self.tables.pop(table_name)

    def _get_table_info_from_query(self, table_name):
        """Get column names and types through query.

        Args:
            table_name (str): Table names

        Returns:
            column_names (list): List of column names
            column_data_types (list): List of column data types
        """
        query = f"PRAGMA table_info({table_name})"
        cursor = self._execute(query)
        columns = cursor.fetchall()
        column_names = []
        column_data_types = []
        for column in columns:
            column_names.append(column[1])
            column_data_types.append(column[2])
        return column_names, column_data_types

    def _delete_all_tables(self):
        """Delete all tables in db file."""
        for table_name in self._load_table_names():
            self._delete_table(table_name)

    def _load_tables(self):
        """Query table info from database."""
        for table_name in self._load_table_names():
            column_names, column_data_types = self._get_table_info_from_query(table_name)
            self.tables[table_name] = dict(zip(column_names, column_data_types))

    def _clean_database(self):
        """Deletes or reloads database from a previous queens run."""
        if self.database_path.is_file():
            if self.reset_existing_db:
                # Reset the database for this run
                self._delete_all_tables()
            else:
                # Load from an existing db file
                self._load_tables()

    def _delete_database(self):
        """Delete database file."""
        self._disconnect()

    def _add_table(self, table_name):
        """Add table if it does not exist.

        Args:
            table_name (str): Table name

        Returns:
            boolean: True if table already existed, False if not
        """
        if table_name in self.tables:
            return self.tables[table_name]
        query = f"SELECT count(name) FROM sqlite_master WHERE type='table' AND name='{table_name}';"
        cursor = self._execute(query)
        counter = cursor.fetchone()[0]
        if counter == 0:
            self.tables.update({table_name: {}})
            self._create_table(table_name)
            return False

    def _create_table(self, table_name):
        """Create an empty table.

        Args:
            table_name (str): table_name
        """
        query = f"CREATE TABLE IF NOT EXISTS {table_name} (id integer PRIMARY KEY)"
        self._execute(query, commit=True)

    def _get_table_column_names(self, table_name):
        """Get names of columns in a table.

        Args:
            table_name (str): Name of the table

        Returns:
            list: names of the column
        """
        return list(self.tables[table_name].keys())

    def _add_column(self, table_name, column_names, column_types):
        """Add columns if necessary.

        Args:
            table_name (str): Name of the table
            column_names (list): List of column names
            column_types (list): List of data types for the columns
        """
        current_column_names = self._get_table_column_names(table_name)
        for i, column_name in enumerate(column_names):
            if not column_name in current_column_names and column_name:
                column_type = column_types[i]
                self.tables[table_name].update({column_name: column_type})
                if column_name != "id":
                    query = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
                    self._execute(query, commit=True)

    def save(self, dictionary, experiment_name, experiment_field, batch, field_filters=None):
        """Save a document to the database.

        Args:
            dictionary (dict):          document to be saved to the db
            experiment_name (string):   experiment the data belongs to
            experiment_field (string):  experiment field data belongs to
            batch (int):                batch the data belongs to
            field_filters (dict):       filter to find appropriate document
                                        to create or update
        Returns:
            bool: is this the result of an acknowledged write operation ?
        """
        table_name = experiment_field
        self._add_table(table_name)
        data_dictionary = dictionary.copy()
        data_dictionary.update({"batch": batch})
        column_names = list(data_dictionary.keys())
        values = list(data_dictionary.values())
        data_types = list(type_to_sqlite(item) for item in values)
        self._add_column(table_name, column_names, data_types)
        if len(column_names) == 1:
            placeholder = "?"
            joint_keys = column_names[0]
        else:
            placeholder = "?," * len(column_names)
            placeholder = placeholder[:-1]
            joint_keys = ", ".join(list(column_names))

        fields_setter = [f"{column_name}=?" for column_name in column_names]
        query = f"INSERT INTO {table_name}"
        query += f" ({joint_keys}) VALUES ({placeholder})"
        fields_setter = [f"{column_name}=excluded.{column_name}" for column_name in column_names]
        query += f" ON CONFLICT(id) DO UPDATE SET {', '.join(fields_setter)}"
        if field_filters is None:
            query += " WHERE id=excluded.id"
        else:
            filter_conditions = [
                f"{filter_column}=?" for filter_column, filter_item in field_filters.items()
            ]
            query += f" WHERE {' AND '.join(filter_conditions)}"
            values += tuple(field_filters.values())
        self._execute(query, parameters=values, commit=True)

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
        table_name = experiment_field
        if self._add_table(table_name):
            column_names = self.tables[table_name].keys()
            query = f"SELECT {', '.join(column_names)} FROM {table_name} WHERE BATCH={batch}"
            if field_filters is not None:
                filter_conditions = [
                    f"{filter_column}='{filter_item}'"
                    for filter_column, filter_item in field_filters.items()
                ]
                query += f" AND {' AND '.join(filter_conditions)}"
            cursor = self._execute(query)
            entries = cursor.fetchall()
            entries = self._list_of_entry_dicts(table_name, entries)
            if len(entries) == 1:
                entries = entries[0]
            return entries
        else:
            return None

    def _list_of_entry_dicts(self, table_name, entries):
        """Create a dict with the column names.

        Args:
            table_name (str): Table name
            entries (list): List obtained from the sqlite api

        Returns:
            list: List of entries as dicts
        """
        column_names = self._get_table_column_names(table_name)
        list_of_dict_entries = []
        for entry in entries:
            entry_dict = dict(zip(column_names, entry))
            entry_dict.pop("batch")
            list_of_dict_entries.append(entry_dict)
        return list_of_dict_entries

    def remove(self, experiment_name, experiment_field, batch, field_filters):
        """Remove a list of documents from the database.

        Args:
            experiment_name (string):  experiment the data belongs to
            experiment_field (string): experiment field data belongs to
            batch (int):               batch the data belongs to
            field_filters (dict):      filter to find appropriate document(s)
                                       to delete
        """
        if field_filters:
            table_name = experiment_field
            field_filters.update({"batch": batch})
            filter_conditions = [
                f"{filter_column}='{filter_item}'"
                for filter_column, filter_item in field_filters.items()
            ]
            query = f"DELETE FROM {table_name} WHERE {' AND '.join(filter_conditions)};"
            self._execute(query, commit=True)
        else:
            raise QUEENSDatabaseError("No field filters for which the data should be removed!")

    def __str__(self):
        """String function of the sqlite object.

        Returns:
            str: table with information
        """
        print_dict = {
            "Name": self.db_name,
            "File": self.database_path.resolve(),
            "Reset db": self.reset_existing_db,
        }
        table = get_str_table("QUEENS SQLite database object wrapper", print_dict)
        return table

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
        table_name = experiment_field

        if self._add_table(table_name):
            query = f"SELECT COUNT(*) FROM {table_name} WHERE BATCH={batch}"
            if field_filters is not None:
                filter_conditions = [
                    f"{filter_column}='{filter_item}'"
                    for filter_column, filter_item in field_filters.items()
                ]
                query += f" AND {' AND '.join(filter_conditions)}"
            cursor = self._execute(query)
            entries = cursor.fetchall()[0][0]
            return entries
        else:
            return 0
