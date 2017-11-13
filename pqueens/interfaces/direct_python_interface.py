import numpy as np
import importlib.util
from .interface import Interface
from pqueens.variables.variables import Variables

class DirectPythonInterface(Interface):
    """ Class for mapping input variables to responses using a python function

        The DirectPythonInterface class maps input variables to outputs,
        i.e. responses by making direct calls to a python function. The function
        has to be defined in a file, which is passed as an argument at runtime.
        The structure of the file must adhere to the structure of the files
        in the folder pqueens/example_input_files. In fact the purppose of
        this class is to be able to call the test examples in said folder.

    Attributes:
        name (string):                  name of interface
        variables (dict):               dictionary with variables
        function (function object):     adress of database to use

    """

    def __init__(self, interface_name, function_file, variables):
        """ Create interface

        Args:
            interface_name (string):    name of interface
            function_file (string):     function file name (including path)
                                        to be executed
            variables (dict):           dictionary with variables

        """
        self.name = interface_name
        self.variables = variables
        # import function using importlib
        spec = importlib.util.spec_from_file_location("my_function", function_file)
        my_function = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(my_function)
        self.function = my_function

    @classmethod
    def from_config_create_interface(cls, interface_name, config):
        """ Create interface from config dictionary

        Args:
            interface_name (str):   name of interface
            config(dict):           dictionary containing problem description

        Returns:
            interface:              instance of DirectPythonInterface
        """
        interface_options = config[interface_name]

        function_file = interface_options["main_file"]
        parameters = config['parameters']

        # instanciate object
        return cls(interface_name, function_file, parameters)

    def map(self, samples):
        """ Mapping function which orchestrates call to simulator function

            First variant of map function with a generator as arguments, which
            can be looped over.

        Args:
            samples (list):         list of variables objects

        Returns:
            np.array,np.array       two arrays containing the inputs from the
                                    suggester, as well as the corresponding outputs
        """
        outputs = []
        for variables in samples:
            params = variables.get_active_variables()
            outputs.append(self.function.main(1, params))
        return outputs
