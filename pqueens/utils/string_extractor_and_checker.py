"""
Module supplies functions to extract string from output
and to check if string is contained in file.
"""


def extract_string_from_output(search_string, output):
    """ Extractor function to to extract string from output.

    Args:
        search_string (str):    string to be searched for
        output (str):           output string
    """

    newstr1 = ""
    for item in output.split("\n"):
        if search_string in item:
            newstr1 = item.strip()
            break
    string_list = ['"', search_string, '": "']
    replace_string = ''.join(filter(None, string_list))
    newstr2 = newstr1.replace(replace_string, '')
    extract_string = newstr2.replace('",', '')

    return extract_string


def check_if_string_in_file(file_name, string_to_search):
    """ Check if any line in the file contains given string """
    # Open the file in read only mode
    with open(file_name, 'r') as read_obj:
        # Read all lines in the file one by one
        for line in read_obj:
            # For each line, check if line contains the string
            if string_to_search in line:
                return True
    return False
