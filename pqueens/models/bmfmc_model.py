import numpy as np
from pqueens.iterators.iterator import Iterator
import pqueens.utils.pdf_estimation as est
from pqueens.iterators.data_iterator import DataIterator
from pqueens.interfaces.bmfmc_interface import BmfmcInterface
from .model import Model
from .simulation_model import SimulationModel
import scipy.stats as st
from pqueens.variables.variables import Variables
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression as PLS
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
import multiprocessing
from joblib import Parallel, delayed


import pdb


class BMFMCModel(Model):
    """ Bayesian Multi-fidelity class
        Attributes:
            interface (interface):          approximation interface

    """

    def __init__(
        self,
        approximation_settings,
        train_in,
        lfs_train_out,
        hf_train_out,
        lf_mc_in,
        lfs_mc_out,
        subordinate_model,
        subordinate_iterator,
        eval_fit,
        error_measures,
        active_learning,
        features_config,
        predictive_var,
        hf_mc=None,
        BMFMC_reference=False,
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
                                             training data --> will be pickle file
            eval_fit (str):                 How to evaluate goodness of fit
            error_measures (list):          List of error measures to compute

        """
        self.interface = None  # gets initialized after feature space is build
        self.approximation_settings = approximation_settings
        self.subordinate_model = subordinate_model  # this is important for active learning
        self.subordinate_iterator = subordinate_iterator
        self.eval_fit = eval_fit
        self.error_measures = error_measures
        self.train_in = train_in
        self.hf_train_out = hf_train_out
        self.lfs_train_out = lfs_train_out  # TODO in general check for column or row data format
        self.lf_mc_in = lf_mc_in
        self.lfs_mc_out = lfs_mc_out
        self.hf_mc = hf_mc
        self.active_learning = active_learning
        self.features_config = features_config
        self.features_test = None
        self.features_train = None
        self.manifold_train = lfs_train_out  # This is just an initialization
        self.manifold_test = lfs_mc_out
        self.f_mean_pred = None
        self.yhf_var_pred = None
        self.support_pyhf = None
        self.pyhf_mean_vec = None
        self.pyhf_var_vec = None
        # This will be set in feature space method-> dict to know which variables are used in
        # regression model
        self.uncertain_parameters = None
        self.predictive_var = predictive_var
        self.pyhf_mc = None
        self.pyhf_mc_support = None
        self.pylf_mc = None
        self.pylf_mc_support = None
        self.BMFMC_reference = BMFMC_reference
        self.sample_mat = None
        # Super call seems necessary to have access to parent class methods but we set parameters
        # to None first and override in feature space method
        super(BMFMCModel, self).__init__(
            name='bmfmc_model', uncertain_parameters=None, data_flag=True
        )
        # model parameters can be accessed over self.variables as renamed like this in parent class

    @classmethod
    def from_config_create_model(
        cls, config, train_in, lfs_train_out, hf_train_out, lf_mc_in, lfs_mc_out, hf_mc=None
    ):
        """  Create data fit surrogate model from problem description

        Args:
            samples : training input for joint_density model (not MC data)
            output : training output for the joint_density model (not MC data)
            feature

        Returns:
            data_fit_surrogate_model:   Instance of DataFitSurrogateModel 
        """
        # get options
        options = config["method"]['joint_density_approx']  # TODO not needed access direclty
        eval_fit = options["eval_fit"]
        BMFMC_reference = config["method"]["method_options"]["BMFMC_reference"]
        error_measures = options["error_measures"]
        active_learning = config['method']['method_options']['active_learning']
        predictive_var = config['method']['method_options']['predictive_var']
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
            subordinate_iterator = None
        ############# End: Active learning ################################

        approximation_settings = config["method"]["joint_density_approx"]["approx_settings"]
        features_config = config["method"]["joint_density_approx"]["features_config"]

        return cls(
            approximation_settings,
            train_in,
            lfs_train_out,
            hf_train_out,
            lf_mc_in,
            lfs_mc_out,
            subordinate_model,
            subordinate_iterator,
            eval_fit,
            error_measures,
            active_learning,
            features_config,
            predictive_var,
            hf_mc,
            BMFMC_reference,
        )

    # TODO: This needs to be checked--> what should be evaluated surrogate
    #  or hf marginal statistics...
    def evaluate(self):
        """ Evaluate model with current set of variables

        Returns:
            np.array: Results correspoding to current set of variables
        """
        # if self.interface == None:
        # standard BMFMC for reference
        self.interface = BmfmcInterface(self.approximation_settings)
        pyhf_mean_BMFMC = None
        pyhf_var_BMFMC = None

        if self.BMFMC_reference == "True":
            self.build_approximation(approx_case="False")
            self.f_mean_pred, self.yhf_var_pred = self.interface.map(self.lfs_mc_out.T)
            self.compute_pyhf_statistics()
            self.compute_pymc_reference()
            pyhf_mean_BMFMC = self.pyhf_mean_vec
            pyhf_var_BMFMC = self.pyhf_var_vec

        # Manifold over feature space
        self.build_approximation(approx_case="True")
        self.f_mean_pred, self.yhf_var_pred = self.interface.map(
            self.manifold_test.T
        )  # TODO the variables (here) manifold must probably an object from the variable class!
        # run methods for marginal statistics with above results as input
        self.compute_pyhf_statistics()
        self.compute_pymc_reference()
        output = {
            'sample_mat': self.sample_mat,
            'manifold_test': self.manifold_test,
            'f_mean': self.f_mean_pred,
            'y_var': self.yhf_var_pred,
            'pyhf_support': self.support_pyhf,
            'pyhf_mean': self.pyhf_mean_vec,
            'pyhf_var': self.pyhf_var_vec,
            'pyhf_mean_BMFMC': pyhf_mean_BMFMC,
            'pyhf_var_BMFMC': pyhf_var_BMFMC,
            'pylf_mc': self.pylf_mc,
            'pyhf_mc': self.pyhf_mc,
            'pyhf_mc_support': self.pyhf_mc_support,
            'pylf_mc_support': self.pylf_mc_support,
        }

        return output

    def build_approximation(self, approx_case=True):
        """ Build underlying approximation """
        # TODO implement proper active learning with subiterator below
        if self.active_learning == "True":
            self.subordinate_iterator.run()

        # train regression model on the data
        if approx_case == "True":
            self.create_features()
            self.interface.build_approximation(self.manifold_train, self.hf_train_out)
        else:
            self.interface.build_approximation(self.lfs_train_out, self.hf_train_out)

        # TODO below might be wrong error measure
        if self.eval_fit == "kfold":
            error_measures = self.eval_surrogate_accuracy_cv(
                X=X, Y=Y, k_fold=5, measures=self.error_measures
            )
            for measure, error in error_measures.items():
                print("Error {} is:{}".format(measure, error))

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

    def compute_pyhf_statistics(self):
        yhf_min = np.min(self.hf_train_out)
        yhf_max = np.max(self.hf_train_out)
        num_dis = 150
        self.support_pyhf = np.linspace(0.9 * yhf_min, 1.1 * yhf_max, num_dis)
        #### pyhf mean prediction ######
        pyhf_mean_vec = np.zeros(self.support_pyhf.shape)
        for mean, var in zip(self.f_mean_pred, self.yhf_var_pred):
            std = np.sqrt(var)
            pyhf_mean_vec = pyhf_mean_vec + st.norm.pdf(self.support_pyhf, loc=mean, scale=std)
        self.pyhf_mean_vec = 1 / self.f_mean_pred.size * pyhf_mean_vec
        #### pyhf var prediction
        num_samples = 50
        sample_mat = self.interface.approximation.m.posterior_samples_f(
            self.manifold_test, full_cov=True, size=num_samples
        )
        self.sample_mat = np.squeeze(sample_mat)
        if self.predictive_var == "True":
            # sampling to run posterior statistics
            nugget = 0  # 3e-4

            pyhf_var_vec = np.zeros([num_dis, num_samples])
            for sample_it, sample_vec in enumerate(self.sample_mat.T):
                for mean, var in zip(sample_vec.T, self.yhf_var_pred):
                    std = np.sqrt(var)
                    pyhf_var_vec[:, sample_it] = pyhf_var_vec[:, sample_it] + st.norm.pdf(
                        self.support_pyhf, loc=mean, scale=std
                    )

            #        def process(self, sample_it, sample_vec, pyhf_var_vec):
            #            for mean, var in zip(sample_vec.T, self.yhf_var_pred):
            #                std = np.sqrt(var)
            #                out = pyhf_var_vec[:, sample_it] + st.norm.pdf(
            #                    self.support_pyhf, loc=mean, scale=std
            #                )
            #                return out
            #
            #            num_cores = multiprocessing.cpu_count()
            #            sample_mat=self.sample_mat.T

            #            pyhf_var_vec = Parallel(n_jobs=num_cores)(
            #                delayed(process)(self, sample_it, sample_vec, pyhf_var_vec)
            #                for (sample_it, sample_vec) in enumerate(sample_mat)
            #            )

            self.pyhf_var_vec = (
                1 / sample_mat.shape[0] * pyhf_var_vec
            ) ** 2  # Just because the plot takes the root of former implementation

            #    # calculate full posterior covariance matrix for testing points
            #    _, k_post = self.interface.approximation.m.predict_noiseless(
            #        self.manifold_test, full_cov=True
            #    )

            #    # TODO this is a quickfix!
            #    f_mean_pred = self.f_mean_pred[0::5, :]
            #    yhf_var_pred = self.yhf_var_pred[0::5, :]
            #    manifold_test = self.manifold_test[0::5, :]
            #    support_diag = np.vstack([self.support_pyhf, self.support_pyhf])
            #    pyhf_squared = np.zeros([support_diag.shape[1]])

            #    for num, (mean1, var1, inp1) in enumerate(
            #        zip(f_mean_pred, yhf_var_pred, manifold_test)
            #    ):
            #        inner_f_mean = f_mean_pred[num + 1 : :, :]
            #        inner_var = yhf_var_pred[num + 1 : :, :]
            #        inner_manifold = manifold_test[num + 1 : :, :]
            #        # if (num/self.f_mean_pred.shape[0] % 0.05)==0:
            #        print(num / f_mean_pred.shape[0])

            #        for num2, (mean2, var2, inp2) in enumerate(
            #            zip(inner_f_mean, inner_var, inner_manifold)
            #        ):
            #            inp1 = np.atleast_2d(inp1)
            #            inp2 = np.atleast_2d(inp2)
            #            # self.interface.approximation.m.posterior_covariance_between_points(
            #            # X1=inp1,X2=inp2)
            #            covariance = k_post[
            #                num + 1 + num2, num
            #            ]
            #            mean_vec = np.vstack([mean1, mean2])
            #            # TODO we seem to have some singularity issues, maybe normalizing or
            #            #  a nugget term or scaling will help
            #            sigma_mat = np.matrix(
            #                [[var1, covariance], [covariance, var2]], dtype='float'
            #            )
            #            pyhf_squared = pyhf_squared + st.multivariate_normal.pdf(
            #                support_diag.T, mean_vec.squeeze(), sigma_mat.squeeze()
            #            )
            #    self.pyhf_var_vec = (
            #        1 / (f_mean_pred.size ** 2) * pyhf_squared
            #    )  # -(self.pyhf_mean_vec)**2
        else:
            self.pyhf_var_vec = None

    def compute_pymc_reference(self):
        bw = 35  # TODO change that as hard coded -> should be optimized
        bandwidth_hfmc = (np.amax(self.hf_mc) - np.amin(self.hf_mc)) / bw
        self.pyhf_mc, self.pyhf_mc_support = est.estimate_pdf(
            np.atleast_2d(self.hf_mc).T, bandwidth_hfmc
        )
        if self.lfs_train_out.shape[1] < 2:
            self.pylf_mc, self.pylf_mc_support = est.estimate_pdf(
                np.atleast_2d(self.lfs_mc_out), bandwidth_hfmc
            )  # TODO: make this also work for several lfs
        else:
            pass  # as already initialized with None

    def create_features(self):
        # load some configs --> stopping criteria / max, min dimensions, input vars ...
        if self.features_config == "manual":
            idx_vec = 0  # features_config["settings"] #TODO should be changed to input file
            self.features_train = self.train_in[:, idx_vec, None]
            self.features_test = self.lf_mc_in[:, idx_vec, None]
            self.manifold_train = np.hstack([self.lfs_train_out, self.features_train])
            self.manifold_test = np.hstack([self.lfs_mc_out, self.features_test])
        elif self.features_config == "deep":
            self.features_train, self.features_test = self.deep_learning()
            self.manifold_train = np.hstack([self.lfs_train_out, self.features_train])
            self.manifold_test = np.hstack([self.lfs_mc_out, self.features_test])
        elif self.features_config == "pls":
            self.features_train, self.features_test = self.pls()
            self.manifold_train = np.hstack([self.lfs_train_out, self.features_train])
            self.manifold_test = np.hstack([self.lfs_mc_out, self.features_test])
        elif self.features_config == "pca":
            self.features_train, self.features_test = self.pca()
            self.manifold_train = np.hstack([self.lfs_train_out, self.features_train])
            self.manifold_test = np.hstack([self.lfs_mc_out, self.features_test])
        elif self.features_config == "sparse_pca":
            self.features_train, self.features_test = self.sparse_pca()
            self.manifold_train = np.hstack([self.lfs_train_out, self.features_train])
            self.manifold_test = np.hstack([self.lfs_mc_out, self.features_test])
        elif self.features_config == "kernel_pca":
            self.features_train, self.features_test = self.kernel_pca()
            self.manifold_train = np.hstack([self.lfs_train_out, self.features_train])
            self.manifold_test = np.hstack([self.lfs_mc_out, self.features_test])

        elif self.features_config == "None":
            pass
        else:
            raise ValueError('Feature space method specified in input file is unknown!')
        # take the lf mc input and output and learn deep feature dimensions
        # add some kind of error check to stop adding features
        # Add feature dim value to training data set so that numeric value of feature dim
        # corresponds to num value of LF/HF
        ######### Set the variables for the regression after feature space in now known ############
        self.uncertain_parameters = {
            "random_variables": {}
        }  # TODO some kind of dict to know which variables are used

        num_lfs = self.lfs_train_out.shape[1]  # TODO check if this is correct
        # set the random variable for the LFs first
        for counter, value in enumerate(self.manifold_test.T):  # iteratre over all lfs
            if counter < num_lfs - 1:
                key = "LF{}".format(counter)
            else:
                key = "Feat{}".format(counter - num_lfs - 1)

            # self.uncertain_parameters["random_variables"][key]['size'] = my_size
            dummy = {key: {'value': value}}
            self.uncertain_parameters['random_variables'].update(
                dummy
            )  # we assume only 1 column per dim
            # self.uncertain_parameters["random_variables"][key]['type'] = float
            # self.uncertain_parameters["random_variables"][key]['distribution'] = None  #TODO check

        # Append random variables for the feature dimensions
        # (random fields are not necessary so far)

        Model.variables = [
            Variables(self.uncertain_parameters)
        ]  # TODO check if data format of manifold_test is correct

    ######### interface end ##############################################
    def pls(self):
        # preprocessing
        # TODO
        # start the partial-least squares
        pls = PLS(n_components=1)
        # test features
        pls.fit(self.lf_mc_in, self.lfs_mc_out[:, 0])
        features_test = pls.transform(self.lf_mc_in)
        # train features
        features_train = pls.transform(self.train_in)
        return features_train, features_test

    def pca(self):
        # Standardizing the features
        x_standardized = StandardScaler().fit_transform(np.vstack((self.lf_mc_in, self.train_in)))
        x_standardized_test = x_standardized[0 : self.lf_mc_in.shape[0], :]
        x_standardized_train = x_standardized[self.lf_mc_in.shape[0] :, :]
        pca_model = PCA(n_components=1)
        pca_model.fit(x_standardized)
        features_test = pca_model.transform(x_standardized_test)
        features_train = pca_model.transform(x_standardized_train)

        return features_train, features_test

    def sparse_pca(self):
        x_standardized = StandardScaler().fit_transform(np.vstack((self.lf_mc_in, self.train_in)))
        x_standardized_test = x_standardized[0 : self.lf_mc_in.shape[0], :]
        x_standardized_train = x_standardized[self.lf_mc_in.shape[0] :, :]
        pca_model = SparsePCA(n_components=1, n_jobs=4, normalize_components=True)
        pca_model.fit(x_standardized)
        features_test = pca_model.transform(x_standardized_test)
        features_train = pca_model.transform(x_standardized_train)

        return features_train, features_test

    def kernel_pca(self):
        # Standardizing the features
        x_standardized = StandardScaler().fit_transform(np.vstack((self.lf_mc_in, self.train_in)))
        x_standardized_test = x_standardized[0 : self.lf_mc_in.shape[0], :]
        x_standardized_train = x_standardized[self.lf_mc_in.shape[0] :, :]
        pca_model = KernelPCA(n_components=1, kernel='rbf', n_jobs=4)
        pca_model.fit(x_standardized)
        features_test = pca_model.transform(x_standardized_test)
        features_train = pca_model.transform(x_standardized_train)

        return features_train, features_test

    def deep_learning(self):
        # check for method settings to choose right calculation
        # train deep learning with lf mc data
        return features
