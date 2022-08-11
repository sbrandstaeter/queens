# -*- coding: utf-8 -*-
"""Database.

This package in essence contains a module to store data related to
computer experiments in a mongodb based database. The class in the
module provides a convenience layer around certain functions which are
needed in QUEENS to read and write input/output data of simulation
models.
"""
from pqueens.utils.import_utils import get_module_class


def from_config_create_database(config):
    """Create new QUEENS database object from config.

    Args:
        config (dict): Problem configuration

    Returns:
        database (obj): Database object
    """
    valid_types = {"mongodb": [".mongodb", "MongoDB"]}

    db_options = config.get("database")
    db_type = db_options.get("type")
    db_class = get_module_class(db_options, valid_types, db_type)

    database = db_class.from_config_create_database(config)
    return database
