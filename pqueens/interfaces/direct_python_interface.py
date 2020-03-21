import os
import importlib.util
from multiprocessing import Pool
import sys

import numpy as np
from tqdm import tqdm

from .interface import Interface


class DirectPythonInterface(Interface):
    """ Class for mapping input variables to responses using a python function

        The DirectPythonInterface class maps input variables to outputs,
        i.e. responses by making direct calls to a python function. The function
        has to be defined in a file, which is passed as an argument at runtime.
        The structure of the file must adhere to the structure of the files
        in the folder pqueens/example_input_files. In fact the purpose of
        this class is to be able to call the test examples in said folder.

    Attributes:
        name (string):                  name of interface
        variables (dict):               dictionary with variables
        function (function object):     address of database to use

    """

    def __init__(self, interface_name, function_file, variables, num_workers=1):
        """ Create interface

        Args:
            interface_name (string):    name of interface
            function_file (string):     function file name (including path)
                                        to be executed
            variables (dict):           dictionary with variables

        """

        self.name = interface_name
        self.variables = variables

        # get path to queens example simulator functions directory
        function_dir = os.path.join(os.path.dirname(__file__), '..', 'example_simulator_functions')
        abs_function_dir = os.path.abspath(function_dir)
        # join paths intelligently, i.e., if function_file contains an
        # absolute path it will be preserved, otherwise the call below will
        # prepend the absolute path to the example_simulator_functions directory
        abs_function_file = os.path.join(abs_function_dir, function_file)
        try:
            spec = importlib.util.spec_from_file_location("my_function", abs_function_file)
            my_function = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(my_function)
            # we want to be able to import the my_function module
            # by name later:
            sys.modules["my_function"] = my_function
        except FileNotFoundError:
            print('Did not find file locally, trying absolute path')
            raise FileNotFoundError(
                "Could not import specified python function " "file! Fix your config file!"
            )

        self.function = my_function

        # pool needs to be created AFTER my_function module is imported
        # and added to sys.module
        if num_workers > 1:
            print(f"Activating parallel evaluation of samples with {num_workers} workers.\n")
            pool = Pool(processes=num_workers)
        else:
            pool = None

        self.pool = pool

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

        num_workers = interface_options.get('num_workers', 1)
        # instantiate object
        return cls(interface_name, function_file, parameters, num_workers)

    def map(self, samples):
        """ Mapping function which orchestrates call to simulator function

        Args:
            samples (list):         list of Variables objects

        Returns:
            dict: dictionary with
                  key:     value:
                  'mean' | ndarray shape:(samples size, shape_of_response)
        """
        output = {}
        mean_values = []
        job_id = 1
        if self.pool is None:
            for variables in tqdm(samples):
                params = variables.get_active_variables()
                mean_value = np.squeeze(self.function.main(job_id, params))
                if not mean_value.shape:
                    mean_value = np.expand_dims(mean_value, axis=0)
                mean_values.append(mean_value)
        else:
            params_list = [(job_id, variables.get_active_variables()) for variables in samples]

            mean_values = self.pool.starmap(self.function.main, params_list)

            for idx, mean_value in enumerate(mean_values):
                mean_value = np.squeeze(mean_value)
                if not mean_value.shape:
                    mean_value = np.expand_dims(mean_value, axis=0)
                mean_values[idx] = mean_value

        output['mean'] = np.array(mean_values)

        return output
