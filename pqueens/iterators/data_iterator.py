"""Data iterator."""

import logging
import pickle

from pqueens.iterators.iterator import Iterator
from pqueens.utils.process_outputs import process_outputs, write_results

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

    def __init__(self, path_to_data, result_description, global_settings):
        """Initialise data iterator.

        Args:
            path_to_data (string):      Path to pickle file containing data
            result_description (dict):  Description of desired results
            global_settings (dict): Global settings of the QUEENS simulations
        """
        super().__init__(None, global_settings)
        self.samples = None
        self.output = None
        self.eigenfunc = None  # TODO this is an intermediate solution--> see Issue #45
        self.path_to_data = path_to_data
        self.result_description = result_description

    @classmethod
    def from_config_create_iterator(cls, config, iterator_name, model=None):
        """Create data iterator from problem description.

        Args:
            config (dict):       Dictionary with QUEENS problem description
            iterator_name (str): Name of iterator to identify right section
                                 in options dict (optional)
            model (model):       Model to use (optional)

        Returns:
            iterator: DataIterator object
        """
        method_options = config[iterator_name].copy()
        method_options.pop('type')
        global_settings = config["global_settings"]

        return cls(**method_options, global_settings=global_settings)

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
                write_results(
                    results,
                    self.global_settings["output_dir"],
                    self.global_settings["experiment_name"],
                )
        # else:
        _logger.info("Size of inputs %s", self.samples.shape)
        _logger.info("Inputs %s", self.samples)
        _logger.info("Size of outputs %s", self.output['mean'].shape)
        _logger.info("Outputs %s", self.output['mean'])

    def read_pickle_file(self):
        """Read in data from a pickle file.

        Main reason for putting this functionality in a method is to make
        mocking reading input easy for testing.

        Returns:
            np.array, np.array: Two arrays, the first contains input samples,
            the second the corresponding output samples
        """
        data = pickle.load(open(self.path_to_data, "rb"))

        return data
