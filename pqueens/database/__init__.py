# -*- coding: utf-8 -*-
"""Database.

This package in essence contains a module to store data related to
computer experiments in a mongodb based database. The class in the
module provides a convenience layer around certain functions which are
needed in QUEENS to read and write input/ouput data of simulation
models.
"""


def from_config_create_database(config):
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
