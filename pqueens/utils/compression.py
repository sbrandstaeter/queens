import zlib
import numpy as np
import base64


COMPRESS_TYPE = 'compressed array'
def compress_array(a):
    """ Compress numpy array using zlib

    Args:
        a(np.array): array to compress
    Returns:
        dict: compressed array
    """
    return {'ctype'  : COMPRESS_TYPE,
            'shape'  : list(a.shape),
            'value'  : base64.b64encode(zlib.compress(a))}

def decompress_array(a):
    """ Decompress numpy array compressed with compress array using zlib

    Args:
        a(dict): Dict containing compressed array
    Returns:
        (np.array) uncompressed np.array
    """
    return np.fromstring(zlib.decompress(base64.b64decode(a['value']))).reshape(a['shape'])



def compress_nested_container(u_container):
    """ Compress all np.arrays in data container passed either as list or dict

    Args:
        u_container (dict,list): list or dict with data to compress

    Returns:
        (dict,list): list or with where are np.arrays have been compressed
    """
    if isinstance(u_container, dict):
        cdict = {}
        for key, value in u_container.items():
            if isinstance(value, dict) or isinstance(value, list):
                cdict[key] = compress_nested_container(value)
            else:
                if isinstance(value, np.ndarray):
                    cdict[key] = compress_array(value)
                else:
                    cdict[key] = value

        return cdict
    elif isinstance(u_container, list):
        clist = []
        for value in u_container:
            if isinstance(value, dict) or isinstance(value, list):
                clist.append(compress_nested_container(value))
            else:
                if isinstance(value, np.ndarray):
                    clist.append(compress_array(value))
                else:
                    clist.append(value)

        return clist


def decompress_nested_container(c_container):
    """ Decompress all np.arrays in data compressed container passed
        either as list or dict

    Args:
        c_container (dict,list): list or dict with compressed data

    Returns:
        (dict,list): list or with where all np.arrays have been decompressed
    """
    if isinstance(c_container, dict):
        if 'ctype' in c_container and c_container['ctype'] == COMPRESS_TYPE:
            try:
                return decompress_array(c_container)
            except:
                raise Exception('Container does not contain a valid array.')
        else:
            udict = {}
            for key, value in c_container.items():
                if isinstance(value, dict) or isinstance(value, list):
                    udict[key] = decompress_nested_container(value)
                else:
                    udict[key] = value

            return udict
    elif isinstance(c_container, list):
        ulist = []
        for value in c_container:
            if isinstance(value, dict) or isinstance(value, list):
                ulist.append(decompress_nested_container(value))
            else:
                ulist.append(value)

        return ulist
