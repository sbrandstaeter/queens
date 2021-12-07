import abc
import logging
import sys

_logger = logging.getLogger(__name__)

# This construct follows the spirit of singleton design patterns
# Informally: there only exists one database instance
this = sys.modules[__name__]
this.database = None


def from_config_create_database(config):
    """Create a QUEENS DB object from config.

    Args:
        config (dict): Problem configuration
    """

    this.database = Database.from_config_create_database(config)


class Database(metaclass=abc.ABCMeta):
    """QUEENS database base-class.

        This class is implemented such that it can be used in a context framework

        with database_obj:
            do_stuff()

    Attributes:
        database_name (str): Database name
        reset_existing_db (boolean): Flag to reset database
    """

    def __init__(self, db_name, reset_existing_db):
        self.db_name = db_name
        self.reset_existing_db = reset_existing_db

    def __enter__(self):
        """
        'enter'-function in order to use the db objects as a context. This function is called
        prior to entering the context
        In this function:
            1. the connection is established
            2. the database may be resetted

        Returns:
            self
        """
        self._connect()
        self._clean_database()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """'exit'-function in order to use the db objects as a context.

        This function is called at the end of the context in order to close the connection to the
        database.

        The exception as well as traceback arguments are required to implement the `__exit__`
        method, however, we do not use them explicitly.

        Args:
            exception_type: indicates class of exception (e.g. ValueError)
            exception_value: indicates exception instance
            traceback: traceback object
        """
        if exception_type:
            _logger.exception(exception_type(exception_value).with_traceback(traceback))

        self._disconnect()

    @abc.abstractmethod
    def save(self):
        """Save an entry to the database."""
        pass

    @abc.abstractmethod
    def load(self):
        """Load an entry from the database."""
        pass

    @abc.abstractmethod
    def remove(self):
        """Remove an entry from the database."""
        pass

    @abc.abstractmethod
    def _connect():
        """Connect to the database."""
        pass

    @abc.abstractmethod
    def _disconnect():
        """Close connection to the database."""
        pass

    @abc.abstractmethod
    def _delete_database(self):
        """Remove a single database."""
        pass

    @abc.abstractmethod
    def _delete_databases_by_prefix(self):
        """Remove all databases based on a prefix."""
        pass

    @abc.abstractmethod
    def _clean_database(self):
        """Clean up the database prior to a queens run.

        This includes actions like reseting existing databases delete
        all related databases or similar.
        """
        pass

    @classmethod
    def from_config_create_database(_, config):
        """Create new QUEENS database object from config.

        Args:
            config (dict): Problem configuration

        Returns:
            Database object
        """

        db_type = config["database"].get("type")

        from .mongodb import MongoDB

        valid_options = {"mongodb": MongoDB}

        if db_type in valid_options.keys():
            return valid_options[db_type].from_config_create_database(config)
        else:
            raise KeyError(
                f"Database type '{db_type}' unknown, valid options are {list(valid_options.keys())}"
            )
