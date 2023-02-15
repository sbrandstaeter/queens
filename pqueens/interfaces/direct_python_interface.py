"""Class for mapping input variables to responses using a python function."""

import numpy as np
from tqdm import tqdm

from pqueens.interfaces.interface import Interface
from pqueens.tests.integration_tests.example_simulator_functions import (
    example_simulator_function_by_name,
)
from pqueens.utils.import_utils import get_module_attribute
from pqueens.utils.pool_utils import create_pool


class DirectPythonInterface(Interface):
    """Class for mapping input variables to responses using a python function.

    The *DirectPythonInterface* class maps input variables to outputs,
    i.e. responses, by making direct calls to a python function. The function
    has to be defined in a file, which is passed as an argument at runtime.
    The structure of the file must adhere to the structure of the files
    in the folder *pqueens/example_input_files*. In fact the purpose of
    this class is to be able to call the test examples in the said folder.

    Attributes:
        function (function):    Function to evaluate.
        pool (pathos pool):     Multiprocessing pool.
        latest_job_id (int):    Latest job ID.
    """

    def __init__(self, interface_name, function, pool):
        """Create interface.

        Args:
            interface_name (string):    name of interface
            function (function):        function to evaluate
            pool (pathos pool):         multiprocessing pool
        """
        super().__init__(interface_name)
        # Wrap function to clean the output
        self.function = self.function_wrapper(function)
        self.pool = pool
        self.latest_job_id = 1

    @classmethod
    def from_config_create_interface(cls, interface_name, config):
        """Create interface from config dictionary.

        Args:
            interface_name (str):   Name of interface
            config(dict):           Dictionary containing problem description

        Returns:
            interface: Instance of DirectPythonInterface
        """
        interface_options = config[interface_name]

        num_workers = interface_options.get('num_workers', 1)
        function_name = interface_options.get("function", None)
        external_python_function = interface_options.get("external_python_module_function", None)

        if function_name is None:
            raise ValueError(f"Keyword 'function' is missing in interface '{interface_name}'")

        if external_python_function is None:
            # Try to load existing simulator functions
            my_function = example_simulator_function_by_name(function_name)
        else:
            # Try to load external simulator functions
            my_function = get_module_attribute(external_python_function, function_name)

        pool = create_pool(num_workers)

        return cls(interface_name=interface_name, function=my_function, pool=pool)

    def evaluate(self, samples):
        """Orchestrate call to simulator function.

        Args:
            samples (list): List of variables objects

        Returns:
            dict: dictionary with
                +----------+------------------------------------------------+
                |**key:**  |  **value:**                                    |
                +----------+------------------------------------------------+
                |'mean'    | ndarray shape (samples size, shape_of_response)|
                +----------+------------------------------------------------+
        """
        number_of_samples = len(samples)

        # List of global sample ids
        sample_ids = np.arange(self.latest_job_id, self.latest_job_id + number_of_samples)

        # Update the latest job id
        self.latest_job_id = self.latest_job_id + number_of_samples

        # Create samples list and add job_id to the dicts
        samples_list = []
        for job_id, sample in zip(sample_ids, samples):
            sample_dict = self.parameters.sample_as_dict(sample)
            sample_dict.update({"job_id": job_id})
            samples_list.append(sample_dict)

        # Pool or no pool
        if self.pool:
            results = self.pool.map(self.function, samples_list)
        else:
            results = list(map(self.function, tqdm(samples_list)))

        output = {}
        # check if gradient is returned --> tuple
        if isinstance(results[0], tuple):
            results_iterator, gradient_iterator = zip(*results)
            results_array = np.array(list(results_iterator))
            gradients_array = np.array(list(gradient_iterator))
            output["gradient"] = gradients_array
        else:
            results_array = np.array(results)

        output["mean"] = results_array
        return output

    @staticmethod
    def function_wrapper(function):
        """Wrap the function to be used.

        This wrapper calls the function by a kwargs dict only and reshapes the output as needed.
        This way if called in a pool, the reshaping is also done by the workers.

        Args:
            function (function): Function to be wrapped

        Returns:
            reshaped_output_function (function): Wrapped function
        """
        pass

        def reshaped_output_function(sample_dict):
            """Call function and reshape output.

            Args:
                sample_dict (dict): Dictionary containing parameters and `job_id`

            Returns:
                (np.ndarray): Result of the function call
            """
            result_array = function(**sample_dict)
            if isinstance(result_array, tuple):
                # here we expect a gradient return
                result = result_array[0]
                gradient = np.array(result_array[1])
                if not result.shape:
                    result = np.expand_dims(result, axis=0)
                    gradient = np.expand_dims(gradient, axis=0)
                return result, gradient
            else:
                # here no gradient return
                if not result_array.shape:
                    result_array = np.expand_dims(result_array, axis=0)
                return result_array

        return reshaped_output_function
