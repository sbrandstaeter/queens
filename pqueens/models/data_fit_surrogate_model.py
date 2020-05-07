import numpy as np
from pqueens.iterators.iterator import Iterator
from pqueens.interfaces.interface import Interface
from .model import Model


class DataFitSurrogateModel(Model):
    """ Surrogate model class

        Attributes:
            interface (interface):          approximation interface

    """

    def __init__(
        self,
        model_name,
        interface,
        model_parameters,
        subordinate_model,
        subordinate_iterator,
        eval_fit,
        error_measures,
    ):
        """ Initialize data fit surrogate model

        Args:
            model_name (string):        Name of model
            interface (interface):      Interface to simulator
            model_parameters (dict):    Dictionary with description of
                                        model parameters
            subordinate_model (model):  Model the surrogate is based on
            subordinate_iterator (Iterator): Iterator to evaluate the subordinate
                                             model with the purpose of getting
                                             training data
            eval_fit (str):                 How to evaluate goodness of fit
            error_measures (list):          List of error measures to compute

        """
        super(DataFitSurrogateModel, self).__init__(model_name, model_parameters)
        self.interface = interface
        # TODO remove this property, as this is not needed
        self.subordinate_model = subordinate_model
        self.subordinate_iterator = subordinate_iterator
        self.eval_fit = eval_fit
        self.error_measures = error_measures

    @classmethod
    def from_config_create_model(cls, model_name, config):
        """  Create data fit surrogate model from problem description

        Args:
            model_name (string): Name of model
            config (dict):       Dictionary containing problem description

        Returns:
            data_fit_surrogate_model:   Instance of DataFitSurrogateModel 
        """
        # get options
        model_options = config[model_name]
        interface_name = model_options["interface"]
        parameters = model_options["parameters"]
        model_parameters = config[parameters]

        subordinate_model_name = model_options["subordinate_model"]
        subordinate_iterator_name = model_options["subordinate_iterator"]
        eval_fit = model_options.get("eval_fit", None)
        error_measures = model_options.get("error_measures", None)

        # create subordinate model
        subordinate_model = Model.from_config_create_model(subordinate_model_name, config)

        # create subordinate iterator
        subordinate_iterator = Iterator.from_config_create_iterator(
            config, subordinate_iterator_name, subordinate_model
        )

        # create interface
        interface = Interface.from_config_create_interface(interface_name, config)

        return cls(
            model_name,
            interface,
            model_parameters,
            subordinate_model,
            subordinate_iterator,
            eval_fit,
            error_measures,
        )

    def evaluate(self):
        """ Evaluate model with current set of variables

        Returns:
            np.array: Results correspoding to current set of variables
        """
        if not self.interface.is_initiliazed():
            self.build_approximation()

        self.response = self.interface.map(self.variables)
        return self.response

    def build_approximation(self):
        """ Build underlying approximation """

        self.subordinate_iterator.run()

        # get samples and results
        X = self.subordinate_iterator.samples
        Y = self.subordinate_iterator.output['mean']

        # train regression model on the data
        self.interface.build_approximation(X, Y)

        if self.eval_fit == "kfold":
            error_measures = self.eval_surrogate_accuracy_cv(
                X=X, Y=Y, k_fold=5, measures=self.error_measures
            )
            for measure, error in error_measures.items():
                print("Error {} is:{}".format(measure, error))
        # TODO check that final surrogate is on all points

        # TODO add proper test for error computation on test set
        # if True:
        #     X_train = X[:100,:]
        #     y_train = Y[:100]
        #     self.interface.build_approximation(X_train, y_train)
        #
        #     X_test = X[101:,:]
        #     y_test = Y[101:]
        #     error_measures = self.eval_surrogate_accuracy(X_test,y_test, self.error_measures)
        #     for measure, error in error_measures.items():
        #         print("Error {} is:{}".format(measure,error))

    def eval_surrogate_accuracy(self, X_test, y_test, measures):
        """ Evaluate the accuracy of the surrogate model based on test set

            Evaluate the accuracy of the surogate model using the provided
            error metrics.

            Args:
                X_test (np.array):  Test inputs
                y_test (np.array):  Test outputs
                measures (list):    List with desired error metrics

            Returns:
                dict: Dictionary with proving error metrics
        """
        if not self.interface.is_initiliazed():
            raise RuntimeError("Cannot compute accuracy on unitialized model")
        X_test_var = self.convert_array_to_model_variables(X_test)
        response = self.interface.map(X_test_var)
        y_response = np.reshape(np.array(response), (-1, 1))
        error_info = self.compute_error_measures(y_test, y_response, measures)
        return error_info

    def eval_surrogate_accuracy_cv(self, X, Y, k_fold, measures):
        """ Compute k-fold cross-validation error

            Args:
                X (np.array):       Input array
                Y (np.array):       Output array
                k_fold (int):       Split dataset in k_fold subsets for cv
                measures (list):    List with desired error metrics

            Returns:
                dict: Dictionary with error measures and correspoding error values
        """
        if not self.interface.is_initiliazed():
            raise RuntimeError("Cannot compute accuracy on unitialized model")

        response_cv = self.interface.cross_validate(X, Y, k_fold)
        y_pred = np.reshape(np.array(response_cv), (-1, 1))

        error_info = self.compute_error_measures(Y, y_pred, measures)
        return error_info

    def compute_error_measures(self, y_act, y_pred, measures):
        """ Compute error measures

            Compute based on difference between predicted and actual values.

            Args:
                y_act (np.array):  Actual values
                y_pred (np.array): Predicted values
                measures (list):   Dictionary with desired error measures

            Returns:
                dict: Dictionary with error measures and correspoding error values
        """
        error_measures = {}
        for measure in measures:
            error_measures[measure] = self.compute_error(y_act, y_pred, measure)
        return error_measures

    def compute_error(self, y_act, y_pred, measure):
        """ Compute error for given a specific error measure

            Args:
                y_act (np.array):  Actual values
                y_pred (np.array): Predicted values
                measure (str):     Desired error metric

            Returns:
                float: error based on desired metric

            Raises:

        """
        # TODO checkout raises field
        if measure == "sum_squared":
            error = np.sum((y_act - y_pred) ** 2)
        elif measure == "mean_squared":
            error = np.mean((y_act - y_pred) ** 2)
        elif measure == "root_mean_squared":
            error = np.sqrt(np.mean((y_act - y_pred) ** 2))
        elif measure == "sum_abs":
            error = np.sum(np.abs(y_act - y_pred))
        elif measure == "mean_abs":
            error = np.mean(np.abs(y_act - y_pred))
        elif measure == "abs_max":
            error = np.max(np.abs(y_act - y_pred))
        else:
            raise NotImplementedError("Desired error measure is unknown!")
        return error
