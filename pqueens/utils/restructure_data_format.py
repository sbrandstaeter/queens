"""Data formatter for databases."""
import io
import pickle

import numpy as np
import pandas as pd

COMPRESS_TYPE = 'uncompressed array'


def convert_array_to_db_dict(numpy_array):
    """Convert numpy arrays in a dictionary format that can be understood by
    MongoDb.

    Args:
        numpy_array (np.array): array to compress

    Returns:
        dict: mongo db compatible dictionary
    """
    return {'ctype': COMPRESS_TYPE, 'shape': list(numpy_array.shape), 'value': numpy_array.tolist()}


def convert_db_dict_to_array(db_dict):
    """Convert a dictionary in a MongoDb compatible format to a numpy array.

    Args:
        db_dict (dict): Dict containing array

    Returns:
        (np.array) np.array
    """
    value = db_dict["value"]
    shape = db_dict["shape"]

    array = np.array(value)

    if list(array.shape) != shape:
        raise ValueError("Error while decompressing the array. ")

    return np.array(value)


def convert_nested_data_to_db_dict(u_container):
    """Restructure nested input data formats into dictionaries that are
    compatible with the MongoDB.

    Args:
        u_container (dict,list): list or dict with data to compress

    Returns:
        (dict,list): list or dict in to MongoDb compatible structure
    """
    if isinstance(u_container, dict):
        cdict = {}
        for key, value in u_container.items():

            # call method recursive in case another dict is encountered
            if isinstance(value, dict) or isinstance(value, list):
                cdict[key] = convert_nested_data_to_db_dict(value)

            # convert np.array to compatible dict
            else:
                if isinstance(value, np.ndarray):
                    cdict[key] = convert_array_to_db_dict(value)
                else:
                    cdict[key] = value

        return cdict

    elif isinstance(u_container, list):
        clist = []
        for value in u_container:
            if isinstance(value, dict) or isinstance(value, list):
                clist.append(convert_nested_data_to_db_dict(value))
            else:
                if isinstance(value, np.ndarray):
                    clist.append(convert_array_to_db_dict(value))
                else:
                    clist.append(value)

        return clist


def convert_nested_db_dicts_to_lists_or_arrays(db_data):
    """Restructure nested dictionaries in the MongoDb compatible format to
    return the original input format of either list or dict type.

    Args:
        db_data (dict,list): dict or list in MongoDb compatible format

    Returns:
        (dict,list): list or dict in original data format or structure
    """
    if isinstance(db_data, dict):
        if 'ctype' in db_data and db_data['ctype'] == COMPRESS_TYPE:
            try:
                return convert_db_dict_to_array(db_data)
            except:
                raise Exception('Container does not contain a valid array.')
        else:
            udict = {}
            for key, value in db_data.items():
                if isinstance(value, dict) or isinstance(value, list):
                    udict[key] = convert_nested_db_dicts_to_lists_or_arrays(value)
                else:
                    udict[key] = value

            return udict
    elif isinstance(db_data, list):
        ulist = []
        for value in db_data:
            if isinstance(value, dict) or isinstance(value, list):
                ulist.append(convert_nested_db_dicts_to_lists_or_arrays(value))
            else:
                ulist.append(value)

        return ulist


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

    Args:
        binarized_bool (bytes): binarized bool

    Returns:
        bool: decoded bool
    """
    return bool(binarized_boolean)
