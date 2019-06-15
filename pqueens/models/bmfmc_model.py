import numpy as np
from pqueens.iterators.iterator import Iterator
from pqueens.iterators.data_iterator import DataIterator
from pqueens.interfaces.interface import Interface
from . model import Model
from . simulation_model import SimulationModel

class BMFMCModel(Model):
    """ Bayesian Multi-fidelity class
        Attributes:
            interface (interface):          approximation interface

    """

    def __init__(self, interface, train_in, lfs_train_out, hf_train_out, lf_mc_in, lfs_mc_out, subordinate_model,
                 subordinate_iterator, eval_fit, error_measures, active_learning, features_config):
        """ Initialize data fit surrogate model

        Args:
            model_name (string):        Name of model
            interface (interface):      Interface to simulator
            model_parameters (dict):    Dictionary with description of
                                        model parameters
            subordinate_model (model):  Model the surrogate is based on
            subordinate_iterator (Iterator): Iterator to evaluate the subordinate
                                             model with the purpose of getting
                                             training data --> will be pickle file
            eval_fit (str):                 How to evaluate goodness of fit
            error_measures (list):          List of error measures to compute

        """
        self.interface = interface
        self.subordinate_model = subordinate_model # this is important for active learning
        self.subordinate_iterator = subordinate_iterator # this is the initial design pickle file (active learning is implemented in model)
        self.eval_fit = eval_fit
        self.error_measures = error_measures
        self.train_in = train_in
        self.hf_train_out = hf_train_out
        self.lfs_train_out = lfs_train_out
        self.lf_mc_in = lf_mc_in
        self.lfs_mc_out = lfs_mc_out
        self.active_learning_flag = active_learning
        self.features_config = features_config
        #model parameters can be accessed over self.variables as renamed like this in parent class

    @classmethod
    def from_config_create_model(cls, config, train_in, lfs_train_out, hf_train_out, lf_mc_in, lfs_mc_out):
        """  Create data fit surrogate model from problem description

        Args:
            samples : training input for joint_density model (not MC data)
            output : training output for the joint_density model (not MC data)
            feature

        Returns:
            data_fit_surrogate_model:   Instance of DataFitSurrogateModel 
        """
        # get options
        options = config['joint_density_approx'] #TODO not needed access direclty
        eval_fit = options["eval_fit"]
        error_measures = options["error_measures"]
        active_learning = config['method']['method_options']['active_learning']
############## Active learning #############################
        if config['method']['method_options']['active_learning'] == "True":
            # TODO: iterator to iterate on joint density (below is just old stuff)
            result_description = None
            global_settings = config.get("global_settings", None)
            subordinate_iterator = DataIterator(path_to_data, result_description, global_settings)
            # TODO: create subordinate model for active learning
            subordinate_model_name = model_options["subordinate_model"]
            subordinate_model = SimulationModel.from_config_create_model(subordinate_model_name,
        else:
            subordinate_model = None
############# End: Active learning ################################
        # create interface for the bmfmc model (not the job interface for active learning)
        interface = Interface.from_config_create_interface('bmfmc_interface', config)
        features_config = config["joint_density_approx"]["features_config"]

        return cls(interface, train_in, lfs_train_out, hf_train_out, lf_mc_in, lfs_mc_out, subordinate_model,
                   subordinate_iterator, eval_fit, error_measures, active_learning, features_config)

    #TODO: This needs to be checked--> what should be evaluated surrogate or hf marginal statistics...
    def evaluate(self):
        """ Evaluate model with current set of variables

        Returns:
            np.array: Results correspoding to current set of variables
        """
        if not self.interface.is_initiliazed():
            self.build_approximation()

        # variables are basically all LF MC runs (not GP LF training points) that need to be made available below
        # run trained model on all LF points and possible features at x to get mean and variane of GP on these points
        joint_mean_vec, joint_var_vec = self.interface.map(self.variables) #TODO variables need to be changed to LF MC
        # run methods for marginal statistics with above results as input
        pyhf_mean_vec = self.compute_pyhf_mean(self.variables,joint_mean_vec,joint_var_vec)
        pyhf_var_vec = self.compute_pyhf_var(self.variables,joint_mean_vec,joint_var_vec)
        # save results in appropriate data format as the "response"
        self.response = #TODO
        return self.response

    def build_approximation(self):
        """ Build underlying approximation """

        #TODO implement proper active learning with subiterator below
         if self.active_learning == "True":
            self.subordinate_iterator.run()

        # train regression model on the data
         self.interface.build_approximation(self.initial_samples, self.initial_output) #TODO: Change in /outputs to appropriate training inputs of jount model !!! --> with features?

        if self.eval_fit == "kfold":
            error_measures = self.eval_surrogate_accuracy_cv(X=X, Y=Y, k_fold=5,
                                                             measures=self.error_measures)
            for measure, error in error_measures.items():
                print("Error {} is:{}".format(measure,error))


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
        # TODO: this needs to be adjusted to fit to the noise uq problem
        # KL divergences and so forth
        if measure == "sum_squared":
            error = np.sum((y_act - y_pred)**2)
        elif measure == "mean_squared":
            error = np.mean((y_act - y_pred)**2)
        elif measure == "root_mean_squared":
            error = np.sqrt(np.mean((y_act - y_pred)**2))
        elif measure == "sum_abs":
            error = np.sum(np.abs(y_act - y_pred))
        elif measure == "mean_abs":
            error = np.mean(np.abs(y_act - y_pred))
        elif measure == "abs_max":
            error = np.max(np.abs(y_act - y_pred))
        else:
            raise NotImplementedError("Desired error measure is unknown!")
        return error

    def compute_pyhf_mean(variables,joint_mean_vec,joint_var_vec):
        pass

    def compute_pyhf_var(variables,joint_mean_vec,joint_var_vec):
        pass

    def create_features(self):
        # load some configs --> stopping criteria / max, min dimensions, input vars ...
        # take the lf mc input and output and learn deep feature dimensions
        # add some kind of error check to stop adding features
        # Add feature dim value to training data set so that numeric value of feature dim corresponds to num value of LF/HF
        pass

