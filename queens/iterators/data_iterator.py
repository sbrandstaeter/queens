"""Data iterator."""

import logging
import pickle

from queens.iterators.iterator import Iterator
from queens.utils.process_outputs import process_outputs, write_results

_logger = logging.getLogger(__name__)


class DataIterator(Iterator):
    """Basic Data Iterator to enable restarts from data.

    Attributes:
        samples (np.array):         Array with all samples.
        output (np.array):          Array with all model outputs.
        eigenfunc: TODO_doc
        path_to_data (string):      Path to pickle file containing data.
        result_description (dict):  Description of desired results.
    """

    def __init__(self, path_to_data, result_description, parameters=None):
        """Initialise data iterator.

        Args:
            path_to_data (string):      Path to pickle file containing data
            result_description (dict):  Description of desired results
            parameters (obj, optional): Parameters
        """
        super().__init__(None, parameters)
        self.samples = None
        self.output = None
        self.eigenfunc = None  # TODO this is an intermediate solution--> see Issue #45
        self.path_to_data = path_to_data
        self.result_description = result_description

    def core_run(self):
        """Read data from file."""
        # TODO: We should return a more general data structure in the future
        # TODO: including I/O and meta data; for now catch it with a try statement
        # TODO: see Issue #45;
        try:
            self.samples, self.output, self.eigenfunc = self.read_pickle_file()
        except ValueError:
            self.samples, self.output = self.read_pickle_file()

    def post_run(self):
        """Analyze the results."""
        if self.result_description is not None:
            results = process_outputs(self.output, self.result_description)
            if self.result_description["write_results"] is True:
                write_results(results, self.output_dir, self.experiment_name)
        # else:
        _logger.info("Size of inputs %s", self.samples.shape)
        _logger.info("Inputs %s", self.samples)
        _logger.info("Size of outputs %s", self.output['result'].shape)
        _logger.info("Outputs %s", self.output['result'])

    def read_pickle_file(self):
        """Read in data from a pickle file.

        Main reason for putting this functionality in a method is to make
        mocking reading input easy for testing.

        Returns:
            np.array, np.array: Two arrays, the first contains input samples,
            the second the corresponding output samples
        """
        with open(self.path_to_data, "rb") as file:
            data = pickle.load(file)

        return data
