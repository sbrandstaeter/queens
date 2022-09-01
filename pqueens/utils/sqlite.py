"""Sqlite utils."""
import io
import pickle
import sqlite3
from functools import partial

import numpy as np
import pandas as pd
import xarray as xr

from pqueens.utils.decorators import safe_operation

# For sqlite the waiting times need to be higher compared to mongodb, especially for fast models
safe_sqlitedb_operation = partial(safe_operation, max_number_of_attempts=10, waiting_time=0.1)


def type_to_sqlite(object_to_be_stored):
    """Get sqlite type from object.

    Args:
        object_to_be_stored: object to be stored in the db

    Returns:
        (str) sqlite data type
    """
    if isinstance(object_to_be_stored, bool):
        return "BOOLEAN"
    if isinstance(object_to_be_stored, str):
        return "TEXT"
    if isinstance(object_to_be_stored, int):
        return "INT"
    if isinstance(object_to_be_stored, float):
        return "REAL"
    if isinstance(object_to_be_stored, xr.DataArray):
        return "XARRAY"
    if isinstance(object_to_be_stored, pd.DataFrame):
        return "PDDATAFRAME"
    if isinstance(object_to_be_stored, np.ndarray):
        return "NPARRAY"
    if isinstance(object_to_be_stored, list):
        return "LIST"
    if isinstance(object_to_be_stored, dict):
        return "DICT"


def sqlite_binary_wrapper(function):
    """Wrap binary output of function to sqlite binary type.

    Args:
        function (fun): Function to be wrapped
    Returns:
        (function) binarized function
    """

    def binarizer(*args, **kwargs):
        binary_out = function(*args, **kwargs)
        return sqlite3.Binary(binary_out)

    return binarizer


@sqlite_binary_wrapper
def np_array_to_binary(np_array):
    """Encode numpy array to binary.

    Args:
        np_array (np.ndarray): Array to be encoded

    Returns:
        bytes: encoded array
    """
    out = io.BytesIO()
    np.save(out, np_array)
    out.seek(0)
    return out.read()


def np_array_from_binary(binarized_array):
    """Decode binary back to numpy array.

    Args:
        binarized_array (bytes): binarized array

    Returns:
        np.ndarray: Decoded array
    """
    out = io.BytesIO(binarized_array)
    out.seek(0)
    return np.load(out)


@sqlite_binary_wrapper
def obj_to_binary(python_object):
    """Encode python object to binary.

    Args:
        python_object (obj): Object to be encoded

    Returns:
        bytes: encoded object
    """
    out = pickle.dumps(python_object, protocol=-1)
    return out


def obj_from_binary(binarized_object):
    """Decode binary back to python object.

    Args:
        binarized_object (bytes): binarized object

    Returns:
        obj: Python object
    """
    return pickle.loads(binarized_object)


@sqlite_binary_wrapper
def pd_dataframe_to_binary(pd_dataframe):
    """Encode dataframe to binary.

    Args:
        pd_dataframe (pd.DataFrame): Dataframe to be encoded

    Returns:
        bytes: encoded dataframe
    """
    out = io.BytesIO()
    pd_dataframe.to_pickle(out)
    out.seek(0)
    return out.read()


def pd_dataframe_from_binary(binarized_dataframe):
    """Decode binary back to pd dataframe.

    Args:
        binarized_dataframe (bytes): binarized object

    Returns:
        pf.dataframe: Dataframe object
    """
    out = io.BytesIO(binarized_dataframe)
    out.seek(0)
    return pd.read_pickle(out)


@sqlite_binary_wrapper
def boolean_to_binary(boolean):
    """Encode bool to binary.

    Args:
        boolean (bool): Bool to be encoded

    Returns:
        bytes: encoded boolean
    """
    return bytes(boolean)


def boolean_from_binary(binarized_boolean):
    """Decode binary back to bool.

    Args
        binarized_bool (bytes): binarized bool

    Returns:
        bool: decoded bool
    """
    return bool(binarized_boolean)
