"""Class for mapping input variables to responses using a python function."""

import numpy as np
from tqdm import tqdm

from pqueens.tests.integration_tests.example_simulator_functions import (
    example_simulator_function_by_name,
)
from pqueens.utils.import_utils import load_main_by_path
from pqueens.utils.pool_utils import create_pool_thread_number

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
        self.function = function
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
        main_file = interface_options.get("main_file", None)
        example_simulator_function = interface_options.get("example_simulator_function", None)
        if main_file is None and example_simulator_function is None:
            raise ValueError(
                f"Please add a main file using the keyword 'main_file' or an example simulator"
                " function with the key 'example_simulator_function' to your input file"
            )
        if main_file is not None and example_simulator_function is not None:
            raise ValueError(
                f"Conflicting inputs: main_file {main_file} and example_simulator_function"
                f"{example_simulator_function}. Only one of these keywords can be set."
            )
        if example_simulator_function:
            my_function = example_simulator_function_by_name(example_simulator_function)
        else:
            my_function = load_main_by_path(main_file)

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
        output = {}
        mean_values = []
        number_of_samples = len(samples)

        # List of global sample ids
        sample_ids = np.arange(self.latest_job_id, self.latest_job_id + number_of_samples)

        # Update the latest job id
        self.latest_job_id = self.latest_job_id + number_of_samples

        # Pool or no pool
        if self.pool is None:
            for job_id, variables in tqdm(zip(sample_ids, samples), total=number_of_samples):
                params = variables.get_active_variables()
                mean_value = np.squeeze(self.function(job_id, params))
                if not mean_value.shape:
                    mean_value = np.expand_dims(mean_value, axis=0)
                mean_values.append(mean_value)
        else:
            params_list = [
                (job_id, variables.get_active_variables())
                for job_id, variables in zip(sample_ids, samples)
            ]

            print(params_list)
            mean_values = self.pool.map(lambda args: self.function(*args), params_list)

            for idx, mean_value in enumerate(mean_values):
                mean_value = np.squeeze(mean_value)
                if not mean_value.shape:
                    mean_value = np.expand_dims(mean_value, axis=0)
                mean_values[idx] = mean_value

        output['mean'] = np.array(mean_values)

        return output
