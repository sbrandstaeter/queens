import numpy as np
import pqueens.utils.pdf_estimation as est
from pqueens.iterators.data_iterator import DataIterator
from pqueens.interfaces.bmfmc_interface import BmfmcInterface
from .model import Model
from .simulation_model import SimulationModel
import scipy.stats as st
from pqueens.variables.variables import Variables
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class BMFMCModel(Model):
    """
    Bayesian multi-fidelity Monte-Carlo model for uncertainty quantification, which is a
    probabilistic mapping between a high-fidelity simulation model (HF) and one or
    more low fidelity simulation models (LFs), respectively informative
    features :math:`(\\gamma)` from the input space. Based on this mapping and the LF samples
    :math:`\\mathcal{D}_{LF}^*=\\{Z^*,Y_{LF}*^\\}`, the BMFMC model computes the
    posterior statistics:

    :math:`\\mathbb{E}_{f^*}\\left[p(y_{HF}^*|f^*,D_f)\\right]`, equation (14) in [1]

    and

    :math:`\\mathbb{V}_{f^*}\\left[p(y_{HF}^*|f^*,D_f)\\right]`, equation (15) in [1]

    of the HF model's output uncertainty.

    The BMFMC model is designed to be constructed upon the sampling data of a LF model
    :math:`\\mathcal{D}_{LF}^*=\\{Z^*,Y_{LF}^*\\}` that are provided by pickle or csv-files,
    and offers than different options to obtain the HF data:

    1.  Provide HF training data in a file (Attention: user needs to make sure that this training
        set is representative and its input :math:`Z` is a subset of the LF model sampling set:
        :math:`Z\\subset Z^*`
    2.  Run optimal HF simulations based on LF data. This requires a suitable simulation sub-model
        and sub-iterator for the HF model. Note: This submodel/iterator can also be used for
        the active learning feature that allows batch-sequential refinement of the BMFMC method by
        determining next optimal HF simulations
    3.  Provide HF sampling data (as a file), calculated with same :math:`Z^*` as the
        LF simulation runs and select optimal HF training set from this data. This is just helpful
        for scientific benchmarking when a ground-truth solution for the HF output uncertainty has
        been sampled before and this data exists anyway.

    Attributes:

        interface (obj): Interface object
        settings_probab_mapping (dict): Settings/configurations for the probabilistic mapping model
                                        between HF and LF models, respectively input features
        subordinate_model (obj): HF (simulation) model to run simulations that yield the HF
                                    training set :math:`\\mathcal{D}_{HF}=\\{Z, Y_{HF}\\}` or HF
                                    model to perform active learning (in to order to extend training
                                    data set of probabilistic mapping with most promising HF
                                    data points)

        eval_fit (str): String that determines which error-evaluation technique should be used to
                        assess the quality of the probabilistic mapping
        error_measures (list): List of string with desired error metrics that should be used to
                               assess the quality of the probabilistic mapping based on
                               cross-validation
        X_train (np.array): Matrix of simulation inputs correspond to the training
                                      data-set of the multi-fidelity mapping
        Y_HF_train (np.array): Vector or matrix of HF output that correspond to training input
                               according to :math:`Y_{HF} = y_{HF}(X)`.
        Y_LFs_train (np.array): Output vector/matrix of one or multiple LF models that correspond to
                                the training input according to :math:`Y_{LF,i}=y_{LF,i}(X)`
        X_mc (np.array): Matrix of simulation inputs that were used in the Monte-Carlo sampling
                         of the LF models. Each row is one input set for a simulation. Columns
                         refer to different realizations of the same variable
        Y_LFs_mc (np.array): Output vector/matrix for the LF models that correspond to the X_mc
                            according to :math:`Y_{LF,i}^*=y_{LF,i}(X^*)`. At the moment Y_LF_mc
                            contains in one row scalar results for different LF models. (In the
                            future we will change the format to pandas dataframes to handle
                            vectorized/functional outputs for different models more elegantly)
        Y_HF_mc (np.array): (optional for benchmarking) Output vector/matrix for the HF model
                            that correspond to the X_mc according to
                            :math:`Y_{HF}^*=y_{HF}(X^*)`.
        active_learning (bool): Flag that triggers active learning on the HF model (not
                                implemented yet)
        features_config (str): String that defines how low-fidelity input features
                               :math:`\\boldsymbol{\\gamma}`should be calculated
        features_mc (np.array): Matrix of low-fidelity informative features
                                :math:`\\boldsymbol{\\Gamma}^*` corresponding to Monte-Carlo
                                input :math:`X^*`
        features_train (np.array): Matrix of low-fidelity informative features
                                   :math:`\\boldsymbol{\\Gamma}` corresponding to the training
                                   input :math:`X`
        Z_train (np.array): Training matrix of low-fidelity features according to
                            :math:`Z=\\left[y_{LF,i}(X),\\Gamma\\right]`
        Z_mc (np.array): Monte-Carlo matrix of low-fidelity features according to
                         :math:`Z^*=\\left[y_{LF,i}(X^*),\\Gamma^*\\right]`
        m_f_mc (np.array): Vector of posterior mean values of multi-fidelity mapping
                           corresponding to the Monte-Carlo input Z_mc according to
                           :math:`\\mathrm{m}_{f^*}(Z^*)`
        var_f_mc (np.array): Vector of posterior variance of multi-fidelity mapping
                             corresponding to the Monte-Carlo input Z_mc according to
                             :math:`\\mathrm{m}_{f^*}(Z^*)`
        y_pdf_support (np.array): Support grid for HF output density :math:`p(y_{HF})`
        p_yhf_mean (np.array): Vector that contains the mean approximation of the HF output
                               density defined on y_hf_support. The vector p_yhf_mean is defined as:
                               :math:`\\mathbb{E}_{f^*}\\left[p(y_{HF}^*|f^*,D_f)\\right]`
                               according to eq. (14) in [1]
        p_yhf_var (np.array): Vector that contains the variance approximation of the HF output
                              density defined on y_hf_support. The vector p_yhf_var is defined as:
                              :math:`\\mathbb{V}_{f^*}\\left[p(y_{HF}^*|f^*,D_f)\\right]`
                              according to eq. (15) in [1]
        predictive_var_bool (bool): Flag that determines whether p_yhf_var should be computed
        p_yhf_mc (np.array): (optional) Monte-Carlo based kernel-density estimate of the HF output
        p_ylf_mc (np.array): (optional) Kernel density estimate for LF model output.
                            Note: For BMFMC the explicit density is never required, only the
                            :math:`\\mathcal{D}_{LF}` is used in the algorithm
        no_features_comparison_bool (bool): If flag is true, the result will be compared to a
                                            prediction that used no LF input features
        num_features (int): Number of informative features of the input :math:`\\boldsymbol{z}_{LF}`
        eigenfunc_random_fields (np.array): Matrix containing the discretized eigenfunctions of a
                                            underlying random field. Note: This is an intermediate
                                            solution and should be moved to the variables module!
                                            The current solution works so far only for one random
                                            field!
        f_mean_train (np.array): Vector of predicted mean values of multi-fidelity mapping
                                 corresponding to the training input Z_train according to
                                 :math:`\\mathrm{m}_{f^*}(Z)`
        lf_data_iterators (obj): Data iterators to load sampling data of low-fidelity models from a
                                 file
        hf_data_iterator (obj):  Data iterator to load the benchmark sampling data from a HF model
                                 from a file (optional and only for scientific benchmark)
    Returns:
        Instance of BMFMCModel

    References:
        [1] Nitzler, J., Biehler, J., Fehn, N., Koutsourelakis, P.-S. and Wall, W.A. (2020),
            "A Generalized Probabilistic  Learning Approach for Multi-Fidelity Uncertainty
            Propagation in Complex Physical Simulations", arXiv:2001.02892
    """

    def __init__(
        self,
        settings_probab_mapping,
        eval_fit,
        error_measures,
        active_learning,
        features_config,
        predictive_var_bool,
        y_pdf_support,
        num_features,
        subordinate_model=None,
        no_features_comparison_bool=False,
        lf_data_iterators=None,
        hf_data_iterator=None,
    ):

        self.interface = None
        self.settings_probab_mapping = settings_probab_mapping
        self.subordinate_model = subordinate_model
        self.eval_fit = eval_fit
        self.error_measures = error_measures
        self.X_train = None
        self.Y_HF_train = None
        self.Y_LFs_train = None
        self.X_mc = None
        self.Y_LFs_mc = None
        self.Y_HF_mc = None
        self.active_learning = active_learning
        self.features_config = features_config
        self.num_features = num_features
        self.features_mc = None
        self.features_train = None
        self.Z_train = None
        self.Z_mc = None
        self.m_f_mc = None
        self.var_f_mc = None
        self.y_pdf_support = None
        self.p_yhf_mean = None
        self.p_yhf_var = None
        self.predictive_var_bool = predictive_var_bool
        self.p_yhf_mc = None
        self.p_ylf_mc = None
        self.no_features_comparison_bool = no_features_comparison_bool
        self.eigenfunc_random_fields = None  # TODO this should be moved to the variable class!
        self.f_mean_train = None
        self.y_pdf_support = y_pdf_support
        self.lf_data_iterators = lf_data_iterators
        self.hf_data_iterator = hf_data_iterator

        super(BMFMCModel, self).__init__(
            name="bmfmc_model", uncertain_parameters=None, data_flag=True
        )  # TODO handling of variables, fields and parameters should be updated!

    @classmethod
    def from_config_create_model(
        cls, model_name, config,
    ):
        """
        Create a BMFMC model from a problem description defined in the input file of QUEENS

        Args:
            config (dict): Dictionary containing the problem description and created from the
                           json-input file
            model_name (str): Name of the model

        Returns:
            BMFMCModel (obj): A BMFMCModel object
        """

        # TODO the unlabeled treatment of raw data for eigenfunc_random_fields and input vars and
        #  random fields is prone to errors and should be changed! The implementation should
        #  rather use the variable module and reconstruct the eigenfunctions of the random fields
        #  if not provided in the data field

        # get model options
        model_options = config['method'][model_name]
        eval_fit = model_options["eval_fit"]
        error_measures = model_options["error_measures"]
        settings_probab_mapping = model_options["approx_settings"]
        features_config = model_options["features_config"]
        lf_data_paths = model_options.get("path_to_lf_data")
        hf_data_path = model_options.get("path_to_hf_data")

        # get some method options
        method_options = config["method"]["method_options"]
        no_features_comparison_bool = method_options["BMFMC_reference"]
        active_learning = method_options["active_learning"]
        model_name = method_options["model"]
        predictive_var_bool = method_options["predictive_var"]
        y_pdf_support_max = method_options["y_pdf_support_max"]
        y_pdf_support_min = method_options["y_pdf_support_min"]

        y_pdf_support = np.linspace(y_pdf_support_min, y_pdf_support_max, 200)
        num_features = config["method"][model_name].get("num_features")
        # ------------------------------ ACTIVE LEARNING ------------------------------
        if active_learning is True:  # TODO also if yhf is not computed yet not only for a.l.
            # TODO: create subordinate model for active learning
            subordinate_HF_model_name = model_options["subordinate_model"]
            subordinate_model = SimulationModel.from_config_create_model(
                subordinate_HF_model_name, config
            )
        else:
            subordinate_model = None  # TODO For now

        # ----------------------- create subordinate data iterators ------------------------------
        lf_data_iterators = [DataIterator(path, None, None) for path in lf_data_paths]
        hf_data_iterator = DataIterator(hf_data_path, None, None)

        return cls(
            settings_probab_mapping,
            eval_fit,
            error_measures,
            active_learning,
            features_config,
            predictive_var_bool,
            y_pdf_support,
            num_features,
            lf_data_iterators=lf_data_iterators,
            hf_data_iterator=hf_data_iterator,
            subordinate_model=subordinate_model,
            no_features_comparison_bool=no_features_comparison_bool,
        )

    def evaluate(self):
        """
        Construct the probabilistic mapping between HF model and LF features and evaluate the
        BMFMC routine. This evaluation consists of two steps.:
            #. Evaluate the probabilistic mapping for LF Monte Carlo Points and the LF training
               points
            #. Use the previous result to actually evaluate the BMFMC posterior statistics

        Returns:
            output (dict): Dictionary containing the core results and some additional quantities:
                           *  Z_mc: LF-features Monte-Carlo data
                           *  m_f_mc: posterior mean values of probabilistic mapping (f) for LF
                                      Monte-Carlo inputs (Y_LF_mc or Z_mc)
                           *  var_f_mc: posterior variance of probabilistic mapping (f) for LF
                                        Monte-Carlo inputs (Y_LF_mc or Z_mc)
                           *  y_pdf_support: PDF support used in this analysis
                           *  p_yhf_mean: Posterior mean prediction of HF output pdf
                           *  p_yhf_var: Posterior variance prediction of HF output pdf
                           *  p_yhf_mean_BMFMC: Reference without features, posterior mean
                                                prediction of HF output pdf
                           *  p_yhf_var_BMMFMC: Reference without features, posterior variance
                                                prediction of HF output pdf
                           *  p_ylf_mc: For illustration purpose, output pdf of LF model
                           *  p_yhf_mc: For benchmarking, output pdf of HF model based on kde
                                        estimate for full Monte-Carlo simulation on HF model
                           *  Z_train: LF feature vector for training of the probabilistic mapping
        """
        self.interface = BmfmcInterface(self.settings_probab_mapping)
        p_yhf_mean_BMFMC = None
        p_yhf_var_BMFMC = None

        if np.any(self.Y_HF_mc):
            self.compute_pymc_reference()

        # ------------------ STANDARD BMFMC (no additional features) for comparison ----------------
        if self.no_features_comparison_bool is True:
            # construct the probabilistic mapping between y_HF and y_LF
            self.build_approximation(approx_case=False)

            # Evaluate probabilistic mapping for LF points
            self.m_f_mc, self.var_f_mc = self.interface.map(self.Y_LFs_mc.T)
            self.f_mean_train, _ = self.interface.map(self.Y_LFs_train.T)

            # actual 'evaluation' of BMFMC routine
            self.compute_pyhf_statistics()
            p_yhf_mean_BMFMC = self.p_yhf_mean  # this is just for comparison so no class attribute
            p_yhf_var_BMFMC = self.p_yhf_var  # this is just for comparison so no class attribute

        # ------------------- Generalized BMFMC with features --------------------------------------
        # construct the probabilistic mapping between y_HF and LF features z_LF
        self.build_approximation(approx_case=True)

        # Evaluate probabilistic mapping for certain Z-points
        self.m_f_mc, self.var_f_mc = self.interface.map(self.Z_mc.T)
        self.f_mean_train, _ = self.interface.map(self.Z_train.T)
        # TODO the variables (here) manifold must probably an object from the variable class!

        # actual 'evaluation' of generalized BMFMC routine
        self.compute_pyhf_statistics()

        # gather and return the output
        output = {
            "Z_mc": self.Z_mc,
            "m_f_mc": self.m_f_mc,
            "var_f_mc": self.var_f_mc,
            "y_pdf_support": self.y_pdf_support,
            "p_yhf_mean": self.p_yhf_mean,
            "p_yhf_var": self.p_yhf_var,
            "p_yhf_mean_BMFMC": p_yhf_mean_BMFMC,
            "p_yhf_var_BMFMC": p_yhf_var_BMFMC,
            "p_ylf_mc": self.p_ylf_mc,
            "p_yhf_mc": self.p_yhf_mc,
            "Z_train": self.Z_train,
        }
        return output

    def load_sampling_data(self):
        """
        Load the low-fidelity sampling data from a pickle file into QUEENS.
        Check if high-fidelity benchmark data is available and load this as well.

        Returns:
            None
        """
        # --------------------- load LF sampling data with data iterators --------------
        self.X_mc = self.lf_data_iterators[0].read_pickle_file()[0]
        # here we assume that all lfs have the same input vector
        try:
            self.eigenfunc_random_fields = self.lf_data_iterators[0].read_pickle_file()[-1]
        except IOError:
            self.eigenfunc_random_fields = None
        Y_LFs_mc = [
            lf_data_iterator.read_pickle_file()[1][:, 0]
            for lf_data_iterator in self.lf_data_iterators
        ]
        self.Y_LFs_mc = np.atleast_2d(np.vstack(Y_LFs_mc)).T

        # ------------------- Deal with potential HF-MC data --------------------------
        if self.hf_data_iterator is not None:
            try:
                _, Y_HF_mc, _ = self.hf_data_iterator.read_pickle_file()
                self.Y_HF_mc = Y_HF_mc[:, 0]  # TODO neglect vectorized output atm
            except FileNotFoundError:
                raise FileNotFoundError(
                    "The file containing the high-fidelity Monte-Carlo data"
                    "was not found! Abort..."
                )
        else:
            raise NotImplementedError(
                "Currently the Monte-Carlo benchmark data for the "
                "high-fidelity model must be provided! In the future QUEENS"
                "will also be able to run the HF simulation based on the"
                "LF data set, automatically. For now abort!...."
            )

    def get_hf_training_data(self):
        """
        Given the low-fidelity sampling data and the optimal training input :math:`X`, either
        simulate the high-fidelity response for :math:`X` or load the corresponding high-fidelity
        response from the high-fidelity benchmark data provided by a pickle file.

        Returns:
            None

        """
        # check if training simulation input was correctly calculated in iterator
        if self.X_train is None:
            raise ValueError(
                "The training input X_train cannot be 'None'! The training inputs "
                "should have been calculated in the iterator! Abort..."
            )

        # check how we should get the corresponding HF simulation output
        if self.Y_HF_mc is not None:
            # match Y_HF_mc data with X_train do determine Y_HF_train
            index_rows = [
                np.where(np.all(self.X_mc == self.X_train[i, :], axis=1))[0][0]
                for i, _ in enumerate(self.X_train[:, 0])
            ]

            self.Y_HF_train = np.atleast_2d(
                np.asarray([self.Y_HF_mc[index] for index in index_rows])
            ).T

        else:
            raise NotImplementedError(
                "Currently the Monte-Carlo benchmark data for the "
                "high-fidelity model must be provided! In the future QUEENS"
                "will also be able to run the HF simulation based on the"
                "LF data set, automatically. For now abort!...."
            )

    def build_approximation(self, approx_case=True):
        """
        Construct the probabilistic surrogate / mapping based on the provided training-data and
        optimize the hyper-parameters by maximizing the data's evidence or its lower bound (ELBO).

        Args:
            approx_case (bool):  Boolean that switches input features :math:`\\boldsymbol{\\gamma}`
                                 off if set to `False`. If not specified or set to `True`
                                 informative input features will be used in the BMFMC framework.

        Returns:
            None
        """

        # get the HF output data (from file or by starting a simulation, dependent on config)
        self.get_hf_training_data()

        # ----- train regression model on the data ----------------------------------------
        if approx_case is True:
            self.set_feature_strategy()
            self.interface.build_approximation(self.Z_train, self.Y_HF_train)
        else:
            self.interface.build_approximation(self.Y_LFs_train, self.Y_HF_train)

        # TODO below might be wrong error measure
        if self.eval_fit == "kfold":
            error_measures = self.eval_surrogate_accuracy_cv(
                self.Z_train, self.Y_HF_train, k_fold=5, measures=self.error_measures
            )
            for measure, error in error_measures.items():
                print("Error {} is:{}".format(measure, error))

        #  TODO implement proper active learning with subiterator below
        if self.active_learning is True:
            raise NotImplementedError(
                'Active learning is not implemented yet! At the moment you '
                'cannot use this option! Please set active_learning to '
                '`False`!'
            )

    def eval_surrogate_accuracy_cv(self, Z, Y_HF, k_fold, measures):
        """
        Compute k-fold cross-validation error for probabilistic mapping

        Args:
            Z (np.array):       Low-fidelity features input-array
            Y_HF (np.array):    High-fidelity output-array
            k_fold (int):       Split dataset in k_fold subsets for cross-validation
            measures (list):    List with desired error metrics

        Returns:
            dict: Dictionary with error metrics and corresponding error values
        """

        if not self.interface.is_initiliazed():
            raise RuntimeError("Cannot compute accuracy of an uninitialized model")

        response_cv = self.interface.cross_validate(Z, Y_HF, k_fold)
        y_pred = np.reshape(np.array(response_cv), (-1, 1))

        error_info = compute_error_measures(Y_HF, y_pred, measures)
        return error_info

    def compute_pyhf_statistics(self):
        """
        Calculate the high-fidelity output density prediction `p_yhf_mean` and its credible bounds
        `p_yhf_var` on the support `y_pdf_support` according to equation (14) and (15) in [1].

        Returns:
            None
        """

        # ---------------------------- PYHF MEAN PREDICTION ---------------------------
        std = np.sqrt(self.var_f_mc)
        pdf_mat = st.norm.pdf(self.y_pdf_support, loc=self.m_f_mc, scale=std)
        pyhf_mean_vec = np.sum(pdf_mat, axis=0)
        self.p_yhf_mean = 1 / self.m_f_mc.size * pyhf_mean_vec
        # ---------------------------- PYHF VAR PREDICTION ----------------------------
        if self.predictive_var_bool:
            # calculate full posterior covariance matrix for testing points
            _, k_post = self.interface.map(self.Z_mc, support='f', full_cov=True)

            # TODO this is a quickfix!
            spacing = 1
            f_mean_pred = self.m_f_mc[0::spacing, :]
            yhf_var_pred = self.var_f_mc[0::spacing, :]
            k_post = k_post[0::spacing, 0::spacing]

            # Define support structure for computation
            points = np.vstack((self.y_pdf_support, self.y_pdf_support)).T

            # Define the outer loop (addition of all multivariate normal distributions
            yhf_pdf_grid = np.zeros((points.shape[0],))
            i = 1
            for num1, (mean1, var1) in enumerate(zip(f_mean_pred, yhf_var_pred)):
                if (num1 % 100) == 0:
                    progress = num1 / f_mean_pred.shape[0] * 100
                    print("Progress variance calculation: %s" % progress)
                for num2, (mean2, var2) in enumerate(
                    zip(f_mean_pred[num1 + 1 :], yhf_var_pred[num1 + 1 :])
                ):
                    #    for num2, (mean2, var2) in enumerate(zip(m_f_mc,var_f_mc)):
                    num2 = num1 + num2
                    covariance = k_post[num1, num2]
                    mean_vec = np.array([mean1, mean2])
                    diff = points - mean_vec.T
                    det_sigma = var1 * var2 - covariance ** 2
                    if det_sigma < 0:
                        det_sigma = 1e-6
                        covariance = 0.95 * covariance
                    inv_sigma = (
                        1
                        / det_sigma
                        * np.array([[var2, -covariance], [-covariance, var1]], dtype=np.float64)
                    )
                    a = np.dot(diff, inv_sigma)
                    b = np.einsum('ij,ij->i', a, diff)
                    c = np.sqrt(4 * np.pi ** 2 * det_sigma)
                    args = -0.5 * b + np.log(1 / c)
                    args[args > 40] = 40
                    yhf_pdf_grid += np.exp(args)
                    i = i + 1

            # Define inner loop (add rows of 2D domain to yield variance function)
            self.p_yhf_var = 1 / (i - 1) * yhf_pdf_grid - 0.9995 * self.p_yhf_mean ** 2
            integral = np.sum(self.p_yhf_var * (self.y_pdf_support[2] - self.y_pdf_support[1]))
            print("intvar=%s" % integral)
        #            with open('cylinder_int_var.txt','a') as myfile:
        #                myfile.write('%s\n' % integral)

        else:
            self.p_yhf_var = None

    def compute_pymc_reference(self):
        """
         Given a high-fidelity Monte-Carlo benchmark dataset, compute the reference kernel
         density estimate for the quantity of interest and optimize the bandwith of the kde.

        Returns:
            None

        """
        # optimize the bandwidth for the kde
        bandwidth_hfmc = est.estimate_bandwidth_for_kde(
            self.Y_HF_mc, np.amin(self.Y_HF_mc), np.amax(self.Y_HF_mc)
        )
        # perform kde with the optimized bandwidth
        self.p_yhf_mc, _ = est.estimate_pdf(
            np.atleast_2d(self.Y_HF_mc),
            bandwidth_hfmc,
            support_points=np.atleast_2d(self.y_pdf_support),
        )
        if self.Y_LFs_train.shape[1] < 2:
            self.p_ylf_mc, _ = est.estimate_pdf(
                np.atleast_2d(self.Y_LFs_mc).T,
                bandwidth_hfmc,
                support_points=np.atleast_2d(self.y_pdf_support),
            )  # TODO: make this also work for several lfs

    def set_feature_strategy(self):
        """
        Depending on the method specified in the input file, set the strategy that will be used to
        calculate the low-fidelity features :math:`Z_{\\text{LF}}`.

        Returns:
            None

        """
        if self.features_config == "manual":
            idx_vec = 0  # TODO should be changed to input file
            self.features_train = self.X_train[:, idx_vec, None]
            self.features_mc = self.X_mc[:, idx_vec, None]
            self.Z_train = np.hstack([self.Y_LFs_train, self.features_train])
            self.Z_mc = np.hstack([self.Y_LFs_mc, self.features_mc])
        elif self.features_config == "pca_joint_space":
            if self.num_features < 1:
                raise ValueError(
                    f'You specified {self.num_features} features, which is an '
                    f'invalid value! Please only specify integer values greater than zero! Abort...'
                )
            self.calculate_z_lf()
        elif self.features_config == "None":
            self.Z_train = self.Y_LFs_train
            self.Z_mc = self.Y_LFs_mc
        else:
            raise ValueError("Feature space method specified in input file is unknown!")

        # TODO current workaround to update variables object with the inputs for the
        #  multi-fidelity mapping
        update_model_variables(self.Y_LFs_train, self.Z_mc)

    def calculate_z_lf(self):
        """
        Given the low-fidelity sampling data, calculate the low-fidelity features
        :math:`z_{\\text{LF}}` based on equation (19) and (20) in [1]. The informative input
        features :math:`\\boldsymbol{\\gamma}` are calculated so that
        they would maximize the Pearson correlation coefficient between :math:`\\gamma_i^*` and
        :math:`Y_{\\text{LF}}^*`. Afterwards :math:`z_{\\text{LF}}` is composed by
        :math:`y_{\\text{LF}}` and :math:`\\boldsymbol{\\gamma_{\\text{LF}}`

        Returns:
            None

        """
        x_standardized_train, x_standardized_test = self.pca()
        x_iter_train = x_standardized_train
        x_iter_test = x_standardized_test
        self.features_train = np.empty((x_iter_train.shape[0], 0))
        self.features_mc = np.empty((x_iter_test.shape[0], 0))

        idx_max = []
        for counter in range(self.num_features):
            ele = np.arange(1, x_iter_train.shape[1] + 1)
            width = 0.25
            plt.rcParams["mathtext.fontset"] = "cm"
            plt.rcParams.update({'font.size': 23})

            fig, ax1 = plt.subplots()

            # y_standardized = StandardScaler().
            # fit_transform((self.Y_HF_train-self.f_mean_train)**2)
            y_standardized2 = StandardScaler().fit_transform(self.Y_LFs_mc)
            #            y_standardized3 = StandardScaler().fit_transform(self.Y_LFs_mc)
            # y_standardized4 = StandardScaler().fit_transform(self.Y_HF_train)
            # y_standardized5 = StandardScaler().\
            # fit_transform((self.Y_HF_mc[:,None]-self.m_f_mc)**2)
            # y_standardized6 = StandardScaler().fit_transform((self.Y_HF_mc[:,None]))
            # Joint space projection
            # inner_proj = np.abs(StandardScaler().\
            # fit_transform(np.dot(x_iter_train.T, y_standardized)))
            inner_proj2 = np.abs(
                StandardScaler().fit_transform(np.dot(x_iter_test.T, y_standardized2))
            )
            # inner_proj3 = np.abs(StandardScaler().\
            # fit_transform(np.dot(x_iter_test.T, y_standardized3)))
            # inner_proj4 = np.abs(StandardScaler().\
            # fit_transform(np.dot(x_iter_train.T, y_standardized4)))
            # inner_proj5 = np.abs(StandardScaler().\
            # fit_transform(np.dot(x_iter_test.T, y_standardized5)))
            # inner_proj6 = np.abs(StandardScaler().\
            # fit_transform(np.dot(x_iter_test.T, y_standardized6)))

            score = inner_proj2[:, 0]
            score[idx_max] = 0

            ax1.bar(ele + width, inner_proj2[:, 0], width, label='ylf', color='g')
            #            ax1.bar(ele,inner_proj4[:,0],width,label='yhf')
            # ax1.bar(ele-width,inner_proj6[:,0],width,label='yhf_full')
            #
            inner_proj = inner_proj2
            # plt.plot(inner_proj, label='comb')
            ax1.grid(which='major', linestyle='-')
            ax1.grid(which='minor', linestyle='--', alpha=0.5)
            ax1.minorticks_on()
            ax1.set_xlabel('Feature')
            ax1.set_ylabel(r'Projection $\mathbf{t}$')
            ax1.set_xticks(ele)
            plt.legend()
            fig.set_size_inches(15, 15)
            path = '/home/nitzler/Documents/Vorlagen/inner_projection_{}.png'.format(counter)
            plt.savefig(path, format='png', dpi=300)

            #   plt.scatter(self.m_f_mc, self.Y_HF_mc[:,None]-self.m_f_mc, label='hferr')
            #   plt.legend()
            #   plt.show()
            select_bool = inner_proj == np.max(
                inner_proj
            )  # alternatively test the error projection of LF
            #            select_bool = inner_proj
            idx_max.append(np.argmax(inner_proj))

            #        # eigendecomp
            #        lamb, v = np.linalg.eig(joint_cov)
            #        lamb = lamb.real
            #        v = v.real
            #
            #        # sort eigenvectors accoring to their eigenvalues
            #        idx = lamb.argsort()[::-1]
            #        lamb = lamb[idx]
            #        v = v[:, idx]

            # select the input
            # test_iter = np.atleast_2d(x_iter_test[:,select_bool.squeeze()])#v[0:num_features]))
            # train_iter = np.atleast_2d(x_iter_train[:,select_bool.squeeze()])# v[0:num_features]))

            test_iter = np.dot(x_iter_test, select_bool)  # v[0:num_features]))
            train_iter = np.dot(
                x_iter_train, select_bool
            )  # v[0:num_features]))            # Rescale the features to y_lf data
            min_ylf = np.min(self.Y_LFs_mc)
            max_ylf = np.max(self.Y_LFs_mc)

            features_train = min_ylf + (train_iter - np.min(train_iter)) * (
                (max_ylf - min_ylf) / (np.max(train_iter) - np.min(train_iter))
            )
            features_test = min_ylf + (test_iter - np.min(test_iter)) * (
                (max_ylf - min_ylf) / (np.max(test_iter) - np.min(test_iter))
            )
            self.features_train = np.hstack((self.features_train, features_train))
            self.features_mc = np.hstack((self.features_mc, features_test))

            # update available inputs
            # x_iter_train = x_iter_train[:, np.logical_not(select_bool).squeeze()]
            # x_iter_test = x_iter_test[:, np.logical_not(select_bool).squeeze()]

            self.Z_train = np.hstack([self.Y_LFs_train, self.features_train])
            self.Z_mc = np.hstack([self.Y_LFs_mc, self.features_mc])

            # update regression model
            self.interface.build_approximation(self.Z_train, self.Y_HF_train)
            self.m_f_mc, self.var_f_mc = self.interface.map(self.Z_mc.T)
            self.f_mean_train, _ = self.interface.map(self.Z_train.T)

    def pca(self):
        # TODO Depreciated method that is currently just used to organize some stuff and will be
        #  replaced in the future

        # Standardizing the features
        x_vec = np.vstack((self.X_mc, self.X_train))

        #        random_fields_test = self.X_mc[:, 1:]  # FSI
        #        random_fields_train = self.X_train[:, 1:] # FSI
        random_fields_test = self.X_mc[:, 3:]  # DG
        random_fields_train = self.X_train[:, 3:]  # DG

        # PCA makes only sense on correlated data set ->
        # seperate correlated and uncorrelated variables
        # TODO this split is hard coded!
        #        x_uncorr = x_vec[:, 0, None]  # FSI
        x_uncorr = x_vec[:, 0:3]  # DG

        x_uncorr_test = x_uncorr[0 : self.X_mc.shape[0], :]
        x_uncorr_train = x_uncorr[self.X_mc.shape[0] :, :]

        # pca_model = PCA(n_components=1) #KernelPCA(n_components=2,
        # kernel="rbf", gamma=10, n_jobs=2)
        # SparsePCA(n_components=3, n_jobs=4, normalize_components=True)
        #        x_trans = pca_model.fit_transform(x_corr)

        # ------------------------- TAKE CARE OF RANDOM FIELDS (DG)  ------------------------
        x_vec = np.linspace(0, 1, 200, endpoint=True)
        mean_fun = 4 * 1.5 * (-((x_vec - 0.5) ** 2) + 0.25)
        normalized_train = random_fields_train - mean_fun
        normalized_test = random_fields_test - mean_fun

        num_trunc = 10  # -1  # 6
        #      pls = PLS(n_components = num_trunc)
        #      pls.fit(random_fields_test,self.Y_LFs_mc)
        # coef_train = pls.transform(random_fields_train).T#
        coef_train = np.dot(self.eigenfunc_random_fields.T, normalized_train.T)[0:num_trunc, :]
        # coef_train = np.linalg.solve(self.eigenfunc_random_fields.T,
        # normalized_train.T)[0:num_trunc,:]
        # coef_test = pls.transform(random_fields_test).T#
        coef_test = np.dot(self.eigenfunc_random_fields.T, normalized_test.T)[0:num_trunc, :]
        # coef_test = np.linalg.solve(self.eigenfunc_random_fields.T,
        # normalized_test.T)[0:num_trunc,:]

        self.eigenfunc_random_fields = self.eigenfunc_random_fields[:, 0:num_trunc]

        # approx = (np.dot(coef_train.T[0:3, :], self.eigenfunc_random_fields.T) + mean_fun).T
        # approx = (np.dot(pls.x_weights_, coef_train))
        # ----------------------- END TAKE CARE OF RANDOM FIELDS (DG)  ----------------------

        ## ---------------------------- RANDOM FIELD FOR FSI ---------------------------
        #        coef_train = random_fields_train.T
        #        coef_test = random_fields_test.T
        ## -------------------------- END RANDOM FIELD FOR FSI -------------------------

        # stack together uncorrelated vars and pca of corr vars
        X_test = np.hstack((x_uncorr_test, coef_test.T))
        X_train = np.hstack((x_uncorr_train, coef_train.T))

        scaler = StandardScaler()
        features_test = scaler.fit_transform(X_test)
        features_train = scaler.transform(X_train)

        return features_train, features_test


# --------------------------- functions ------------------------------------------------------
def compute_error(y_act, y_pred, measure):
    """ Compute error for a given a specific error metric

        Args:
            y_act (np.array):  Prediction with full data set
            y_pred (np.array): Prediction with reduced data set
            measure (str):     Desired error metric

        Returns:
            float: error based on desired metric
    """
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


def compute_error_measures(y_act, y_pred, measures):
    """
        Compute error-metrics based on difference between prediction with full data-set and reduced
        data-set

        Args:
            y_act (np.array):  Predictions with full data-set
            y_pred (np.array): Predictions with reduced data-set
            measures (list):   Dictionary with desired error metrics

        Returns:
            dict: Dictionary with error measures and corresponding error values
    """
    error_measures = {}
    for measure in measures:
        error_measures[measure] = compute_error(y_act, y_pred, measure)
    return error_measures


def update_model_variables(Y_LFs_train, Z_mc):
    """
    Intermediate solution: Update the QUEENS variable object with the previous calculated
    low-fidelity features :math:`Z_{\\text{LF}}`

    Args:
        Y_LFs_train (np.array): Low-fidelity outputs :math:`Y_{\\text{LF}}` for training input
                                :math:`X`.
        Z_mc (np.array): Low-fidelity feature matrix :math:`Z_{\\text{LF}}^{*}` corresponding to
        sampling input :math:`X^{*}`

    Returns:
        None
    """
    # TODO this is an intermediate solution while the variable class has not been changed to a
    #  more flexible version

    uncertain_parameters = {
        "random_variables": {}
    }  # initialize a dict uncertain parameters to define input_variables of model

    num_lfs = Y_LFs_train.shape[1]  # TODO not a very nice solution but work for now

    # set the random variable for the LFs first
    for counter, value in enumerate(Z_mc.T):  # iterate over all lfs
        if counter < num_lfs - 1:
            key = "LF{}".format(counter)
        else:
            key = "Feat{}".format(counter - num_lfs - 1)

        dummy = {key: {"value": value}}
        uncertain_parameters["random_variables"].update(dummy)  # we assume only 1 column per dim

    # Append random variables for the feature dimensions (random fields are not necessary so far)
    Model.variables = [Variables(uncertain_parameters)]
