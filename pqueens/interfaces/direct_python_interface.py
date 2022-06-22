"""Class for mapping input variables to responses using a python function."""

import numpy as np
from tqdm import tqdm

from pqueens.tests.integration_tests.example_simulator_functions import (
    example_simulator_function_by_name,
)
from pqueens.utils.import_utils import load_function_by_name_from_path
from pqueens.utils.pool_utils import create_pool_thread_number
from pqueens.utils.valid_options_utils import InvalidOptionError

from .interface import Interface


class DirectPythonInterface(Interface):
    """Class for mapping input variables to responses using a python function.

        The DirectPythonInterface class maps input variables to outputs,
        i.e. responses by making direct calls to a python function. The function
        has to be defined in a file, which is passed as an argument at runtime.
        The structure of the file must adhere to the structure of the files
        in the folder pqueens/example_input_files. In fact the purpose of
        this class is to be able to call the test examples in said folder.

    Attributes:
        name (string):          name of interface
        variables (dict):       dictionary with variables
        function (function):    function to evaluate
        pool (pathos pool):     multiprocessing pool
        latest_job_id (int):    Latest job id
    """

    def __init__(self, interface_name, function, variables, pool):
        """Create interface.

        Args:
            interface_name (string):    name of interface
            function (function):        function to evaluate
            variables (dict):           dictionary with variables
            pool (pathos pool):         multiprocessing pool
        """
        self.name = interface_name
        self.variables = variables
        # Wrap function to clean the output
        self.function = self.function_wrapper(function)
        self.pool = pool
        self.latest_job_id = 1

    @classmethod
    def from_config_create_interface(cls, interface_name, config, driver_name):
        """Create interface from config dictionary.

        Args:
            interface_name (str):   name of interface
            config(dict):           dictionary containing problem description
            driver_name (str): Name of the driver that uses this interface
                               (not used here)

        Returns:
            interface:              instance of DirectPythonInterface
        """
        interface_options = config[interface_name]

        parameters = config['parameters']

        num_workers = interface_options.get('num_workers', 1)
        function_name = interface_options.get("function_name", None)
        external_python_module = interface_options.get("external_python_module", None)

        if external_python_module is None:
            try:
                my_function = example_simulator_function_by_name(function_name)
            except InvalidOptionError:
                # Could not find the function in example simulator function of QUEENS
                pass
        else:
            my_function = load_function_by_name_from_path(external_python_module, function_name)

        pool = create_pool_thread_number(num_workers)

        return cls(
            interface_name=interface_name, function=my_function, variables=parameters, pool=pool
        )

    def evaluate(self, samples):
        """Mapping function which orchestrates call to simulator function.

        Args:
            samples (list):         list of Variables objects

        Returns:
            dict: dictionary with
                  key:     value:
                  'mean' | ndarray shape:(samples size, shape_of_response)
        """
        number_of_samples = len(samples)

        # List of global sample ids
        sample_ids = np.arange(self.latest_job_id, self.latest_job_id + number_of_samples)

        # Update the latest job id
        self.latest_job_id = self.latest_job_id + number_of_samples

        # Create samples list and add job_id to the dicts
        samples_list = []
        for job_id, variables in zip(sample_ids, samples):
            sample_dict = variables.get_active_variables()
            sample_dict.update({"job_id": job_id})
            samples_list.append(sample_dict)

        # Pool or no pool
        if self.pool:
            results = self.pool.map(self.function, samples_list)
        else:
            results = list(map(self.function, tqdm(samples_list)))

        output = {'mean': np.array(results)}

        return output

    @staticmethod
    def function_wrapper(function):
        """Wrap the function to be used.

        This wrapper calls the function by a kwargs dict only and reshapes output as needed. This way if called in a pool the reshaping is also done by the workers.

        Args:
            function (function): function to be wrapped

        Returns:
            reshaped_output_function (function): wrapped function
        """

        def reshaped_output_function(sample_dict):
            """Call function and reshape output

            Args:
                sample_dict (dict): dictionary containing parameters and `job_id`

            Returns:
                (np.ndarray): result of the function call
            """
            result = function(**sample_dict)
            result = np.squeeze(result)
            if not result.shape:
                result = np.expand_dims(result, axis=0)
            return result

        return reshaped_output_function
