"""
Extractor module supplies function to extract string from AWS output.
"""


def aws_extract(search_string, output):
    """ Extractor function to to extract string from AWS output.

    Args:
        search_string (str):    string to be searched for
        output (str):           AWS output string
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
