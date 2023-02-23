"""DummyDB module."""
import logging

from pqueens.database.database import Database

_logger = logging.getLogger(__name__)


class DummyDB(Database):
    """DummyDB used if no DB is wanted."""

    def __init__(
        self,
    ):
        """Initialize dummydb object.

        Returns:
            DummyDB (obj): Instance of MongoDB class
        """
        db_name = "DummyDB"
        reset_existing_db = False
        super().__init__(db_name, reset_existing_db)

    @classmethod
    def from_config_create_database(cls, config):
        """Create Mongo database object from problem description.

        Args:
            config (dict): Dictionary containing the problem description of the current QUEENS
                           simulation

        Returns:
            MongoDB (obj): Instance of MongoDB class
        """
        return cls()

    def _connect(self):
        """Connect to the database."""

    def _clean_database(self):
        """Reset the database."""

    def _disconnect(self):
        """Close connection."""

    def _delete_database(self):
        """Delete database."""

    def save(self):
        """Save to database."""

    def remove(self):
        """Remove from database."""

    def load(self):
        """Load from database."""

    def __str__(self):
        """Represen DummyDB as string."""
        return "\nRunning without database.\n"
