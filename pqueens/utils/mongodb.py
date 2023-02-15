"""MongoDB utils."""
import numpy as np

COMPRESS_TYPE = 'uncompressed array'


def convert_array_to_db_dict(numpy_array):
    """Convert numpy arrays in a dictionary format for MongoDB.

    Args:
        numpy_array (np.array): Array to compress

    Returns:
        dict: MongoDB compatible dictionary
    """
    return {'ctype': COMPRESS_TYPE, 'shape': list(numpy_array.shape), 'value': numpy_array.tolist()}


def convert_db_dict_to_array(db_dict):
    """Convert a dictionary in a MongoDb compatible format to a numpy array.

    Args:
        db_dict (dict): Dict containing array

    Returns:
        np.array: np.array
    """
    value = db_dict["value"]
    shape = db_dict["shape"]

    array = np.array(value)

    if list(array.shape) != shape:
        raise ValueError("Error while decompressing the array. ")

    return np.array(value)


def convert_nested_data_to_db_dict(u_container):
    """Restructure nested input data into dictionaries for MongoDB.

    Args:
        u_container (dict,list): List or dict with data to compress

    Returns:
        dict,list: List or dict in to MongoDB compatible structure
    """
    if isinstance(u_container, dict):
        cdict = {}
        for key, value in u_container.items():

            # call method recursive in case another dict is encountered
            if isinstance(value, (dict, list)):
                cdict[key] = convert_nested_data_to_db_dict(value)

            # convert np.array to compatible dict
            else:
                if isinstance(value, np.ndarray):
                    cdict[key] = convert_array_to_db_dict(value)
                else:
                    cdict[key] = value

        return cdict

    if isinstance(u_container, list):
        clist = []
        for value in u_container:
            if isinstance(value, (dict, list)):
                clist.append(convert_nested_data_to_db_dict(value))
            else:
                if isinstance(value, np.ndarray):
                    clist.append(convert_array_to_db_dict(value))
                else:
                    clist.append(value)

        return clist


def convert_nested_db_dicts_to_lists_or_arrays(db_data):
    """Restructure nested dictionaries in the MongoDB compatible format.

    Return the original input format of either list or dict type.

    Args:
        db_data (dict,list): Dict or list in MongoDB compatible format

    Returns:
        dict,list: List or dict in original data format or structure
    """
    if isinstance(db_data, dict):
        if 'ctype' in db_data and db_data['ctype'] == COMPRESS_TYPE:
            try:
                return convert_db_dict_to_array(db_data)
            except Exception as exception:
                raise TypeError('Container does not contain a valid array.') from exception
        else:
            udict = {}
            for key, value in db_data.items():
                if isinstance(value, (dict, list)):
                    udict[key] = convert_nested_db_dicts_to_lists_or_arrays(value)
                else:
                    udict[key] = value

            return udict
    elif isinstance(db_data, list):
        ulist = []
        for value in db_data:
            if isinstance(value, (dict, list)):
                ulist.append(convert_nested_db_dicts_to_lists_or_arrays(value))
            else:
                ulist.append(value)

        return ulist


def create_experiment_field_name(driver_name):
    """Create experiment field name from driver_name.

    Args:
        driver_name (str): Name of the current driver

    Return:
        experiment_field_name (str): Name of the experiment field
                                     in the database where current jobs
                                     are saved.
    """
    experiment_field_name = 'jobs_' + driver_name
    return experiment_field_name
