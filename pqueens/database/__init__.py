# -*- coding: utf-8 -*-
"""Database.

This package in essence contains a module to store data related to
computer experiments in a mongodb based database. The class in the
module provides a convenience layer around certain functions which are
needed in QUEENS to read and write input/output data of simulation
models.
"""
from pqueens.utils.import_utils import get_module_class

VALID_TYPES = {"mongodb": ["pqueens.database.mongodb", "MongoDB"]}


def from_config_create_database(config):
    """Create new QUEENS database object from config.

    Args:
        config (dict): Problem configuration

    Returns:
        database (obj): Database object
    """
    db_options = config.get("database")
    db_class = get_module_class(db_options, VALID_TYPES)
    database = db_class.from_config_create_database(config)
    return database
