"""Class for mapping input variables to responses using a python function."""

import numpy as np
from tqdm import tqdm

from queens.example_simulator_functions import example_simulator_function_by_name
from queens.interfaces.interface import Interface
from queens.utils.import_utils import get_module_attribute
from queens.utils.pool_utils import create_pool


class DirectPythonInterface(Interface):
    """Class for mapping input variables to responses using a python function.

    The *DirectPythonInterface* class maps input variables to outputs,
    i.e. responses, by making direct calls to a python function. The function
    has to be defined in a file, which is passed as an argument at runtime.
    The structure of the file must adhere to the structure of the files
    in the folder *queens/example_input_files*. In fact the purpose of
    this class is to be able to call the test examples in the said folder.

    Attributes:
        function (function):    Function to evaluate.
        pool (pathos pool):     Multiprocessing pool.
        verbose (boolean):      Verbosity of evaluations.
    """

    def __init__(
        self,
        parameters,
        function,
        num_workers=1,
        external_python_module_function=None,
        verbose=True,
    ):
        """Create interface.

        Args:
            parameters (obj): Parameters object
            function (callable, str): Function or name of example function provided by QUEENS
            external_python_module_function (pathos pool): Path to external module with function
            num_workers (int): Number of workers
            verbose (boolean): verbosity of evaluations
        """
        super().__init__(parameters)
        if external_python_module_function is None:
            if isinstance(function, str):
                # Try to load existing simulator functions
                my_function = example_simulator_function_by_name(function)
            else:
                my_function = function
        else:
            # Try to load external simulator functions
            my_function = get_module_attribute(external_python_module_function, function)

        pool = create_pool(num_workers)

        # Wrap function to clean the output
        self.function = self.function_wrapper(my_function)
        self.pool = pool
        self.verbose = verbose

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
        samples_list = self.create_samples_list(samples)

        # Pool or no pool
        if self.pool:
            results = self.pool.map(self.function, samples_list)
        elif self.verbose:
            results = list(map(self.function, tqdm(samples_list)))
        else:
            results = list(map(self.function, samples_list))

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
            # here no gradient return
            if not result_array.shape:
                result_array = np.expand_dims(result_array, axis=0)
            return result_array

        return reshaped_output_function
