import pickle
from pqueens.iterators.iterator import Iterator
from pqueens.utils.process_outputs import process_ouputs
from pqueens.utils.process_outputs import write_results


class DataIterator(Iterator):
    """ Basic Data Iterator to enable restarts from data

    Attributes:
        samples (np.array):         Array with all samples
        output (np.array):          Array with all model outputs
        path_to_data (string):      Path to pickle file containing data
        result_description (dict):  Description of desired results

    """

    def __init__(self, path_to_data, result_description, global_settings):
        super(DataIterator, self).__init__(None, global_settings)
        self.samples = None
        self.output = None
        self.eigenfunc = None  # TODO this is an intermediate solution--> see Issue #45
        self.path_to_data = path_to_data
        self.result_description = result_description

    @classmethod
    def from_config_create_iterator(cls, config, iterator_name=None, model=None):
        """ Create data iterator from problem description

        Args:
            config (dict):       Dictionary with QUEENS problem description
            iterator_name (str): Name of iterator to identify right section
                                 in options dict (optional)
            model (model):       Model to use (optional)

        Returns:
            iterator: DataIterator object

        """
        print(config.get("experiment_name"))
        if iterator_name is None:
            method_options = config["method"]["method_options"]
        else:
            method_options = config[iterator_name]["method_options"]

        path_to_data = method_options.get("path_to_data", None)
        result_description = method_options.get("result_description", None)
        global_settings = config.get("global_settings", None)

        return cls(path_to_data, result_description, global_settings)

    def eval_model(self):
        """ Evaluate the model """
        pass

    def pre_run(self):
        pass

    def core_run(self):
        """  Read data from file """
        # TODO: We should return a more general data structure in the future
        # TODO: including I/O and meta data; for now catch it with a try statement
        # TODO: see Issue #45;
        try:
            self.samples, self.output, self.eigenfunc = self.read_pickle_file()
        except ValueError:
            self.samples, self.output = self.read_pickle_file()

    def post_run(self):
        """ Analyze the results """
        if self.result_description is not None:
            results = process_ouputs(self.output, self.result_description)
            if self.result_description["write_results"] is True:
                write_results(
                    results,
                    self.global_settings["output_dir"],
                    self.global_settings["experiment_name"],
                )
        # else:
        print("Size of inputs {}".format(self.samples.shape))
        print("Inputs {}".format(self.samples))
        print("Size of outputs {}".format(self.output['mean'].shape))
        print("Outputs {}".format(self.output['mean']))

    def read_pickle_file(self):
        """ Read in data from a pickle file

        Main reason for putting this functionality in a method is to make
        making mocking reading input easy for testing

        Returns:
            np.array, np.array: Two arrays, the first contains input samples,
                                the second the corresponding output samples

        """

        data = pickle.load(open(self.path_to_data, "rb"))

        samples = data["input"]  # TODO think about an unified format/key, before: "input_data"
        output = data["output"]  # TODO think about an unified format/key, before: "raw_output_data"
        eigenfunc = data.get('eigenfunc')

        return samples, output, eigenfunc
