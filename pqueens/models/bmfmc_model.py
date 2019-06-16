import numpy as np
from pqueens.iterators.iterator import Iterator
from pqueens.iterators.data_iterator import DataIterator
from pqueens.interfaces.bmfmc_interface import BmfmcInterface
from . model import Model
from . simulation_model import SimulationModel
import scipy.stats as st

class BMFMCModel(Model):
    """ Bayesian Multi-fidelity class
        Attributes:
            interface (interface):          approximation interface

    """

    def __init__(self, approximation_settings,train_in, lfs_train_out, hf_train_out, lf_mc_in, lfs_mc_out, subordinate_model,
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
        self.interface = None # gets initialized after feature space is build
        self.approximation_settings = approximation_settings
        self.subordinate_model = subordinate_model # this is important for active learning
        self.subordinate_iterator = subordinate_iterator # this is the initial design pickle file (active learning is implemented in model)
        self.eval_fit = eval_fit
        self.error_measures = error_measures
        self.train_in = train_in
        self.hf_train_out = hf_train_out
        self.lfs_train_out = lfs_train_out #TODO in general check for column or row data format
        self.lf_mc_in = lf_mc_in
        self.lfs_mc_out = lfs_mc_out
        self.active_learning_flag = active_learning
        self.features_config = features_config
        self.features_test = None
        self.features_train = None
        self.manifold_train = lfs_train_out # This is just an initialization
        self.manifold_test = lfs_mc_out
        self.f_mean_pred = None
        self.yhf_var_pred = None
        self.support_pyhf = None
        self.pyhf_mean_vec = None
        self.pyhf_var_vec = None
        self.uncertain_parameters=None # This will be set in feature space method-> dict to know which variables are used in regression model

        super(BMFMCModel, self).__init(name='bmfmc_model',uncertain_parameters=None, data_flag=True) # Super call seems necessary to have access to parent class methods but we set parameters to None first and override in feature space method
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
        if active_learning == "True":
            # TODO: iterator to iterate on joint density (below is just old stuff)
            result_description = None
            global_settings = config.get("global_settings", None)
            subordinate_iterator = DataIterator(path_to_data, result_description, global_settings)
            # TODO: create subordinate model for active learning
            subordinate_model_name = model_options["subordinate_model"]
            subordinate_model = SimulationModel.from_config_create_model(subordinate_model_name)
        else:
            subordinate_model = None
############# End: Active learning ################################

        approximation_settings = config["joint_density_approx"]["approx_settings"]
        features_config = config["joint_density_approx"]["features_config"]

        return cls(approximation_settings,train_in, lfs_train_out, hf_train_out, lf_mc_in, lfs_mc_out, subordinate_model,
                   subordinate_iterator, eval_fit, error_measures, active_learning, features_config)

    #TODO: This needs to be checked--> what should be evaluated surrogate or hf marginal statistics...
    def evaluate(self):
        """ Evaluate model with current set of variables

        Returns:
            np.array: Results correspoding to current set of variables
        """
        if not self.interface.is_initiliazed():
            self.build_approximation()

        self.f_mean_pred, self.yhf_var_pred = self.interface.map(self.manifold_test) #TODO the variables (here) manifold must probably an object from the variable class!
        # run methods for marginal statistics with above results as input
        self.compute_pyhf_statistics()

        return self.f_mean_pred, self.yhf_var_pred # TODO for now we return the hf output and its variacne and return the densities later

    def build_approximation(self):
        """ Build underlying approximation """
        self.create_features()
        #TODO implement proper active learning with subiterator below
        if self.active_learning == "True":
            self.subordinate_iterator.run()

        # train regression model on the data
        self.interface.build_approximation(self.manifold_train, self.hf_train_out) #TODO: Change in /outputs to appropriate training inputs of jount model !!! --> with features?

        # TODO below might be wrong error measure
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

    def compute_pyhf_statistics(self):
        yhf_min = np.min(self.hf_train_out)
        yhf_max = np.max(self.hf_train_out)
        self.support_pyhf = np.linspace(0.5*yhf_min,1.5*yhf_max,300)

        #### pyhf mean prediction ######
        pyhf_mean_vec = np.zeros(self.support_pyhf.shape)
        for (mean,var) in zip(self.f_mean_pred, self.yhf_var_pred):
            pyhf_mean_vec = pyhf_mean_vec + st.norm.pdf(self.support_pyhf,mean,np.sqrt(var))
        self.pyhf_mean_vec = 1/self.f_mean_pred.size * pyhf_mean_vec
        #### pyhf var prediction
        pyhf_squared = np.zeros(self.support_pyhf.shape)
        support_grid = np.meshgrid(self.support_pyhf,self.support_pyhf)
        for (mean1, var1,inp1) in zip(self.f_mean_pred,self.yhf_var_pred,self.manifold_test):
            for (mean2, var2,inp2) in zip(self.f_mean_pred,self.yhf_var_pred,self.manifold_test):
                f_covariance = (inp1,inp2) #TODO get GP posterior covariance fun
                mean_vec = np.vstack([mean1,mean2])
                sigma_mat = np.array([[var1,f_covariance],[f_covariance, var2]])
                pyhf_squared = pyhf_squared + st.multivariate_normal.pdf(support_grid,mean_vec,sigma_mat)
        pyhf_squared_var = np.diag(pyhf_squared)
        self.pyhf_var_vec = 1/self.f_mean_pred.size**2*pyhf_squared_var -(self.pyhf_mean_vec)**2



    def create_features(self):
        # load some configs --> stopping criteria / max, min dimensions, input vars ...
        if self.features_config["method"] == "manual":
            idx_vec = features_config["settings"]
            self.features_train = self.lfs_train_out[:,idx_vec]
            self.features_test = self.lfs_mc_out[:,idx_vec]
            self.manifold_train = np.hstack([self.lfs_train_out, self.features_train])
            self.manifold_test = np.hstack([self.lfs_mc_out, self.features_test])
        elif self.features_config["method"] == "deep":
            self.features_train, self.features_test = self.deep_learning()
            self.manifold_train = np.hstack([self.lfs_train_out, self.features_train])
            self.manifold_test = np.hstack([self.lfs_mc_out, self.features_test])
        elif self.features_config["method"] == "none":
            pass
        else:
            raise ValueError('Feature space method is specified in input file is unknown!')
        # take the lf mc input and output and learn deep feature dimensions
        # add some kind of error check to stop adding features
        # Add feature dim value to training data set so that numeric value of feature dim corresponds to num value of LF/HF
######### Set the variables for the regression after feature space in now known #################
        self.uncertain_parameters = {} #TODO some kind of dict to know which variables are used
        num_lfs = self.lfs_train_out.shape[1] # TODO check if this is correct
        # set the random variable for the LFs first
        for counter, value in enumerate(self.manifold_test.T): # iteratre over all lfs
            if counter < num_lfs-1:
                key = "LF{}".format(counter)
            else:
                key = "".format(counter-num_lfs-1)

            #self.uncertain_parameters["random_variables"][key]['size'] = my_size
            self.uncertain_parameters["random_variables"][key]['value'] = value # we assume only 1 column per dim
            #self.uncertain_parameters["random_variables"][key]['type'] = float
            #self.uncertain_parameters["random_variables"][key]['distribution'] = None  #TODO check

        # Append random variables for the feature dimensions (random fields are not necessary so far)

        Model.variables = [Variables.from_data_vector_create(self.uncertain_paramerters,self.manifold_test)] #TODO check if data format of manifold_test is correct



########## create interfafjafkce for the bmfmc model (not the job interface for active learning) #########
        self.interface = BMFMCInterface(self.approximation_settings)
######### interface end ##############################################

    def deep_learning(self):
        # check for method settings to choose right calculation
        # train deep learning with lf mc data
        return features
