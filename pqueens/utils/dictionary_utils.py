def findkeys(dictionary, key):
    """Get all the values of a 'key' in a nested dictionary.

    Args:
        key ('str'): String of key in dictionary we are interested in
        dictionary (dict): Actual dictionary variable

    Retruns:
        result (list): List of values that have the specified key.
    """
    if isinstance(dictionary, list):
        for i in dictionary:
            for x in findkeys(i, key):
                yield x
    elif isinstance(dictionary, dict):
        if key in dictionary:
            yield dictionary[key]
        for j in dictionary.values():
            for x in findkeys(j, key):
                yield x
