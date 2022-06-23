"""Class for mapping input variables to responses using a python function."""
import importlib.util
import logging
import os
import sys
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from pqueens.utils.path_utils import relative_path_from_pqueens

from .interface import Interface

_logger = logging.getLogger(__name__)


class DirectPythonInterface(Interface):
    """Class for mapping input variables to responses using a python function.

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
        """Create interface.

        Args:
            interface_name (string):    name of interface
            function_file (string):     function file name (including path)
                                        to be executed
            variables (dict):           dictionary with variables
            num_workers (int):          number of worker processes
        """
        self.name = interface_name
        self.variables = variables

        # get path to queens example simulator functions directory
        abs_function_dir = relative_path_from_pqueens(
            'tests/integration_tests/example_simulator_functions'
        )
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
        except FileNotFoundError as my_error:
            raise FileNotFoundError(
                "Could not import specified python function file! Fix your config file!"
            ) from my_error

        self.function = my_function

        # pool needs to be created AFTER my_function module is imported
        # and added to sys.module
        if num_workers > 1:
            _logger.info(f"Activating parallel evaluation of samples with {num_workers} workers.\n")
            pool = Pool(processes=num_workers)
        else:
            pool = None

        self.pool = pool

    @classmethod
    def from_config_create_interface(cls, interface_name, config, **_kargs):
        """Create interface from config dictionary.

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

    def evaluate(self, samples, gradient_bool=False):
        """Mapping function which orchestrates call to simulator function.

        Args:
            samples (list): List of variables objects
            gradient_bool (bool): Flag to determine, whether the gradient of the function at
                                  the evaluation point is expected (True) or not (False)

        Returns:
            dict: dictionary with
                  key:     value:
                  'mean' | ndarray shape:(samples size, shape_of_response)
        """
        output = {}
        mean_values = []
        gradient_values = []
        job_id = 1
        if self.pool is None:
            for variables in tqdm(samples):
                params = variables.get_active_variables()
                model_output = self.function.main(job_id, params)

                mean_values, gradient_values = DirectPythonInterface._get_correct_model_outputs(
                    model_output, mean_values, gradient_values, gradient_bool=gradient_bool
                )

        else:
            params_list = [(job_id, variables.get_active_variables()) for variables in samples]
            model_outputs = self.pool.starmap(self.function.main, params_list)

            for model_output in model_outputs:
                mean_values, gradient_values = DirectPythonInterface._get_correct_model_outputs(
                    model_output, mean_values, gradient_values, gradient_bool=gradient_bool
                )

        output['mean'] = np.array(mean_values)
        output['gradient'] = np.array(gradient_values)
        return output

    @staticmethod
    def _get_correct_model_outputs(model_output, mean_values, gradient_values, gradient_bool=False):
        """Synthesize the correct model outputs."""
        if gradient_bool:
            model_response = np.squeeze(model_output[0])
            model_gradient = np.squeeze(model_output[1])
        else:
            model_response = np.squeeze(model_output)
            model_gradient = np.empty(model_response.shape)

        if not model_response.shape:
            model_response = np.expand_dims(model_response, axis=0)
            model_gradient = np.expand_dims(model_gradient, axis=0)

        mean_values.append(model_response)
        gradient_values.append(model_gradient)

        return mean_values, gradient_values
