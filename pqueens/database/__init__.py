# -*- coding: utf-8 -*-
"""Database.

This package in essence contains a module to store data related to
computer experiments in a mongodb based database. The class in the
module provides a convenience layer around certain functions which are
needed in QUEENS to read and write input/output data of simulation
models.
"""

from pqueens.utils.import_utils import get_module_attribute
from pqueens.utils.valid_options_utils import get_option


def from_config_create_database(config):
    """Create new QUEENS database object from config.

    Args:
        config (dict): Problem configuration

    Returns:
        database (obj): Database object
    """
    from .mongodb import MongoDB

    db_options = config.get("database")
    valid_options = {"mongodb": MongoDB}

    if db_options.get("external_python_module"):
        module_path = db_options["external_python_module"]
        module_attribute = db_options.get("type")
        db_class = get_module_attribute(module_path, module_attribute)
    else:
        db_class = get_option(valid_options, db_options.get("type"))

    database = db_class.from_config_create_database(config)
    return database
