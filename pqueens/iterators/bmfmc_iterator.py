from pqueens.models.bmfmc_model import BMFMCModel
from .iterator import Iterator
from random import randint
from diversipy import *
import pandas as pd
import pqueens.visualization.bmfmc_visualization as qvis
from pqueens.utils.process_outputs import write_results
from pqueens.utils.process_outputs import process_ouputs


class BMFMCIterator(Iterator):
    """
    Iterator for the (generalized) Bayesian multi-fidelity Monte-Carlo method. The iterator
    fulfills the following tasks:

    1.  Load the low-fidelity Monte Carlo data
    2.  Based on low-fidelity data, calculate optimal X_train to evaluate the high-fidelity model
    3.  Based on X_train return the corresponding Y_LFs_train
    4.  Initialize the BMFMC_model (this is not the high-fidelity model but the probabilistic
        mapping) with X_train and Y_LFs_train. Note that the BMFMC_model itself triggers the
        computation of the high-fidelity training data Y_HF_train.
    5.  Trigger the evaluation of the BMFMC_model. Here evaluation refers to computing the
        posterior statistics of the high-fidelity model. This is implemented in the BMFMC_model
        itself.

    Attributes:

        model (obj): Instance of the BMFMCModel
        result_description (dict): Dictionary containing settings for plotting and saving data/
                                   results
        X_train (np.array): Corresponding input for the simulations that are used to train the
                            probabilistic mapping
        Y_HF_train (np.array): Outputs of the high-fidelity model that correspond to the training
                               inputs X_train such that :math:`Y_{HF}=y_{HF}(X)`
        Y_LFs_train (np.array): Outputs of the low-fidelity models that correspond to the training
                                inputs X_train
        eigenfunc_random_fields_train (np.array): (Intermediate solution) Array containing
                                                  eigenfunctions of involved random fields,
                                                  such that 99 % of their variance is covered.
                                                  Eigenfunctions are only required for
                                                  determination of informative features where only
                                                  the first few eigenfunctions are involved so
                                                  that we have very relaxed demands on the
                                                  accuracy of the field reconstruction.
        output (dict): Dictionary containing the output quantities:

                       *  "Z_mc": Corresponding Monte-Carlo point in LF informative feature space
                       *  "m_f_mc": Corresponding Monte-Carlo points of posterior mean of
                                    the probabilistic mapping
                       *  "var_y_mc": Corresponding Monte-Carlo posterior variance samples of the
                                      probabilistic mapping
                       *  "y_pdf_support": Support vector for QoI output distribution
                       *  "p_yhf_mean": Vector containing mean function of HF output
                                        posterior distribution
                       *  "p_yhf_var": Vector containing posterior variance function of HF output
                                       distribution
                       *  "p_yhf_mean_BMFMC": Vector containing mean function of HF output
                                              posterior distribution calculated without informative
                                              features :math:`\\boldsymbol{\\gamma}`
                       *  "p_yhf_var_BMFMC": Vector containing posterior variance function of HF
                                             output distribution calculated without informative
                                             features :math:`\\boldsymbol{\\gamma}`
                       *  "p_ylf_mc": Vector with low-fidelity output distribution (kde from MC
                                      data)
                       *  "p_yhf_mc": Vector with reference HF output distribution (kde from MC
                                      reference data)
                       *  "Z_train": Corresponding training data in LF feature space
                       *  "Y_HF_train": Outputs of the high-fidelity model that correspond to the
                                     training inputs X_train such that :math:`Y_{HF}=y_{HF}(X)`
                       *  "X_train": Corresponding input for the simulations that are used to
                                     train the probabilistic mapping

        initial_design (dict): Dictionary containing settings for the selection strategy/initial
                               design of training points for the probabilistic mapping
        predictive_var (bool): Boolean flag that triggers the computation of the posterior variance
                               :math:`\\mathbb{V}_{f}\\left[p(y_{HF}^*|f,\\mathcal{D})\\right]` if
                               set to True. (default value: False)
        BMFMC_reference (bool): Boolean that triggers the BMFMC solution without informative
                                features :math:`\\boldsymbol{\\gamma}` for comparison if set to
                                True (default
                                value: False)

    Returns:

       BMFMCIterator (obj): Instance of the BMFMCIterator

    """

    def __init__(
        self,
        model,
        result_description,
        initial_design,
        predictive_var,
        BMFMC_reference,
        global_settings,
    ):
        #  TODO check if None for the model is appropriate here
        super(BMFMCIterator, self).__init__(
            None, global_settings
        )  # Input prescribed by iterator.py
        self.model = model
        self.result_description = result_description
        self.X_train = None
        self.Y_LFs_train = None
        self.eigenfunc_random_fields_train = None
        self.output = None
        self.initial_design = initial_design
        self.predictive_var = predictive_var
        self.BMFMC_reference = BMFMC_reference

    @classmethod
    def from_config_create_iterator(cls, config, iterator_name=None, model=None):
        """
        Create LHS iterator from problem description in input file

        Args:
            config (dict): Dictionary with QUEENS problem description
            iterator_name (str): Name of iterator to identify right section in options dict
            model (obj): Instance of model class

        Returns:
            iterator (obj): BMFMCIterator object

        """
        # Initialize Iterator and model
        method_options = config["method"]["method_options"]
        BMFMC_reference = method_options["BMFMC_reference"]
        result_description = method_options["result_description"]
        predictive_var = method_options["predictive_var"]

        initial_design = config["method"]["initial_design"]
        global_settings = config.get('global_settings', None)

        # ----------------------------- CREATE BMFMC MODEL ----------------------------
        if model is None:
            model_name = method_options["model"]
            model = BMFMCModel.from_config_create_model(model_name, config)

        # ---------------------- CREATE VISUALIZATION BORG ----------------------------
        qvis.from_config_create(config)

        return cls(
            model,
            result_description,
            initial_design,
            predictive_var,
            BMFMC_reference,
            global_settings,
        )

    def core_run(self):
        """
        Main run of the BMFMCIterator that covers the following points:

        1.  Reading the sampling data from the low-fidelity model in QUEENS
        2.  Based on LF data, determine optimal X_train for which the high-fidelity model should
            be evaluated :math:`Y_{HF}=y_{HF}(X)`
        3.  Update the BMFMCModel with the partial training data set of X_train, Y_LF_train (
            Y_HF_train is determined in the BMFMCModel)
        4.  Evaluate the BMFMCModel which means that the posterior statistics
            :math:`\\mathbb{E}_{f}\\left[p(y_{HF}^*|f,\\mathcal{D})\\right]` and
            :math:`\\mathbb{V}_{f}\\left[p(y_{HF}^*|f,\\mathcal{D})\\right]` are computed based
            on the BMFMC algorithm which is implemented in the BMFMCModel

        Returns:
            None

        """
        # -------- Load MC data from model -----------------------
        self.model.load_sampling_data()

        # ---- determine optimal input points for which HF should be simulated -------
        self.calculate_optimal_X_train()

        # ----- build model on training points and evaluate it -----------------------
        self.output = self.eval_model()

    def calculate_optimal_X_train(self):
        """
        Based on the low-fidelity sampling data, calculate the optimal model inputs X_train on
        which the high-fidelity model should be evaluated to construct the training data set for
        BMFMC. This selection is performed based on the following method options:

        1. **random**: Divides the :math:`y_{LF}` data set in bins and selects training
                       candidates random from each bin until :math:`n_{train}` is reached
        2. **diverse subset**: Determine the most important input features :math:`\\gamma_i`
                               (this information is provided by the BMFMCModel) and find a space
                               filling subset (diverse subset) given the LF sampling data with
                               respect to the most important features :math:`\\gamma_i`. The
                               number of features to be considered can be set in the input file.
                               **Remark**: An optimization routine for the optimal number of
                               features to be considered will be added in the future

        Returns:
            None

        """
        design_method = self.initial_design.get('method')
        n_points = self.initial_design.get("num_HF_eval")
        run_design_method = self._get_design_method(design_method)
        run_design_method(n_points)

        # update the Bmfmc model variables
        # TODO: normally done by model.update_model_from_sample_batch() !
        self.model.X_train = self.X_train
        self.model.Y_LFs_train = self.Y_LFs_train

    def _get_design_method(self, design_method):
        """
        Get the design method for selecting the HF data from the LF MC data-set

        Args:
            design_method (str): Design method specified in input file

        Returns:
            run_design_method (obj): Design method for selecting the HF training set

        """
        self.model.calculate_extended_gammas()
        if design_method == 'random':
            run_design_method = self._random_design

        elif design_method == 'diverse_subset':
            run_design_method = self._diverse_subset_design

        else:
            raise NotImplementedError(
                f"You specified the non-valid method"
                f" '{self.initial_design['method']}'. This is not "
                f"implemented! The only valid methods are 'random' or "
                f"'diverse_subset'. Abort..."
            )

        return run_design_method

    def _diverse_subset_design(self, n_points):
        """
        Calculate the HF training points from large LF-MC data-set based on the diverse subset
        strategy based on the psa_select method from **diversipy**.

        Args:
             n_points (int): Number of HF training points to be selected

        Returns:
            None

        """
        design = self.model.gammas_ext_mc
        prelim_subset = psa_select(design, n_points, selection_target='max_dist_from_boundary')

        # return training data for outputs and corresponding inputs
        index = np.vstack(
            np.array(
                np.all((design[:, None, :] == prelim_subset[None, :, :]), axis=-1).nonzero()
            ).T.tolist()
        )[:, 0]

        # set the training data and indices in the BMFMC model and iterator
        self.model.training_indices = index
        self.X_train = self.model.X_mc[index, :]
        self.Y_LFs_train = self.model.Y_LFs_mc[index, :]

    def _random_design(self, n_points):
        """
        Calculate the HF training points from large LF-MC data-set based on random selection
        from bins over y_LF.

        Args:
            n_points (int): Number of HF training points to be selected

        Returns:
            None

        """
        n_bins = self.initial_design.get("num_bins")
        seed = self.initial_design.get("seed")
        ylf_min = np.amin(self.model.Y_LFs_mc)
        ylf_max = np.amax(self.model.Y_LFs_mc)
        break_points = np.linspace(ylf_min, ylf_max, n_bins + 1)

        # TODO: bin_vec only works for one LF --> user should define a 'master LF' for
        #  binning at the moment the first LF in the list is taken as the 'master LF'
        bin_vec = pd.cut(
            self.model.Y_LFs_mc[:, 0],
            bins=break_points,
            labels=False,
            include_lowest=True,
            retbins=True,
        )

        # Some initialization
        self.Y_LFs_train = np.empty((0, self.model.Y_LFs_mc.shape[1]))

        self.X_train = np.array([]).reshape(0, self.model.X_mc.shape[1])

        if self.model.eigenfunc_random_fields is not None:
            self.eigenfunc_random_fields_train = np.array([]).reshape(
                0, self.model.eigenfunc_random_fields.shape[1]
            )

        # Go through all bins and  randomly select points
        training_indices = []
        for bin_n in range(n_bins):
            # array of booleans
            y_in_bin_bool = [bin_vec[0] == bin_n]

            # define bin data
            bin_data_X_mc = self.model.X_mc[tuple(y_in_bin_bool)]
            bin_data_Y_LFs_mc = self.model.Y_LFs_mc[tuple(y_in_bin_bool)]

            # randomly select points in bins
            rnd_select = []
            for _ in range(n_points // n_bins):
                random.seed(seed)
                rnd_select.append(randint(0, bin_data_Y_LFs_mc.shape[0] - 1))
                seed += 1

            # Define X_train and Y_LFs_train by checking the bins
            if len(rnd_select) != 0:
                self.X_train = np.vstack([self.X_train, bin_data_X_mc[rnd_select, :]])
                self.Y_LFs_train = np.vstack((self.Y_LFs_train, bin_data_Y_LFs_mc[rnd_select, :]))
            # return training indices to the BMFMC model
            training_indices.extend(rnd_select)

        self.model.training_indices = np.array(training_indices)

    def eval_model(self):
        """
        Evaluate the BMFMCModel which means that the posterior statistics
             :math:`\\mathbb{E}_{f}\\left[p(y_{HF}^*|f,\\mathcal{D})\\right]` and
             :math:`\\mathbb{V}_{f}\\left[p(y_{HF}^*|f,\\mathcal{D})\\right]` are computed based
             on the BMFMC algorithm which is implemented in the BMFMCModel

        Returns:
            None

        """
        return self.model.evaluate()

    # ------------------- BELOW JUST PLOTTING AND SAVING RESULTS ------------------
    def post_run(self):
        """
        Saving and plotting of the results. The latter will be moved to a separate module in the
        future.

        Returns:
            None
        """
        # plot the figures
        qvis.bmfmc_visualization_instance.plot_pdfs(self.output)
        qvis.bmfmc_visualization_instance.plot_manifold(
            self.output, self.model.Y_LFs_mc, self.model.Y_HF_mc, self.model.Y_HF_train
        )

        if self.result_description['write_results'] is True:
            results = process_ouputs(self.output, self.result_description)
            write_results(
                results, self.global_settings["output_dir"], self.global_settings["experiment_name"]
            )
