from pqueens.models.bmfmc_model import BMFMCModel
from .iterator import Iterator
from random import randint
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from diversipy import *
import pandas as pd


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
        experiment_dir (str): Path to the experiment directory where simulations are stored
        X_train (np.array): Corresponding input for the simulations that are used to train the
                            probabilistic mapping
        Y_HF_train (np.array): Outputs of the high-fidelity model that correspond to the training
                               inputs X_train such that :math:`Y_{HF}=y_{HF}(X)`
        Y_LFs_train (np.array): Outputs of the low-fidelity models that correspond to the training
                                inputs X_train
        eigenfunc_random_fields_train (np.array):
        output (dict):
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
        experiment_dir,
        initial_design,
        predictive_var,
        BMFMC_reference,
        global_settings,
    ):

        super(BMFMCIterator, self).__init__(
            None, global_settings
        )  # Input prescribed by iterator.py
        self.model = model
        self.result_description = result_description
        self.experiment_dir = experiment_dir
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
        experiment_dir = method_options["experiment_dir"]
        predictive_var = method_options["predictive_var"]

        initial_design = config["method"]["initial_design"]
        global_settings = config.get('global_settings', None)

        # ----------------------------- CREATE BMFMC MODEL ----------------------------
        if model is None:
            model_name = method_options["model"]
            model = BMFMCModel.from_config_create_model(model_name, config)

        return cls(
            model,
            result_description,
            experiment_dir,
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

        # update the Bmfmc model variables
        # TODO: normally done by model.update_model_from_sample_batch() !
        self.model.X_train = self.X_train
        self.model.Y_LFs_train = self.Y_LFs_train

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
        n_bins = self.initial_design.get("num_bins")
        n_points = self.initial_design.get("num_HF_eval")

        # -------------------------- RANDOM FROM BINS METHOD --------------------------
        if design_method == 'random':
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
            for bin_n in range(n_bins):
                # array of booleans
                y_in_bin_bool = [bin_vec[0] == bin_n]

                # define bin data
                bin_data_X_mc = self.model.X_mc[tuple(y_in_bin_bool)]
                bin_data_Y_LFs_mc = self.model.Y_LFs_mc[tuple(y_in_bin_bool)]

                # randomly select points in bins
                rnd_select = []
                for _ in range(n_points // n_bins):
                    rnd_select.append(randint(0, bin_data_Y_LFs_mc.shape[0] - 1))

                # Define X_train and Y_LFs_train by checking the bins
                if len(rnd_select) != 0:
                    self.X_train = np.vstack([self.X_train, bin_data_X_mc[rnd_select, :]])
                    self.Y_LFs_train = np.vstack(
                        (self.Y_LFs_train, bin_data_Y_LFs_mc[rnd_select, :])
                    )
        # --------------------------- DIVERSE SUBSET METHOD ---------------------------
        elif design_method == 'diverse_subset':
            # TODO handling of random fields is hard coded
            #  --> change this according to issue #87

            # TODO this repeats code form model.set_feature_strategy and should be changed
            x_vec = np.linspace(0, 1, 200, endpoint=True)  # DG
            mean_fun = 4 * 1.5 * (-((x_vec - 0.5) ** 2) + 0.25)  # DG
            normalized_test = self.model.X_mc[:, 3:] - mean_fun  # DG
            coef_test = np.dot(self.model.eigenfunc_random_fields.T, normalized_test.T).T  # DG
            design = np.hstack((self.model.X_mc[:, 0:3], coef_test[:, 0:2]))  # DG

            # design = np.hstack((self.model.X_mc[:, 0:1], self.model.Y_LFs_mc[:]))  # FSI

            # ------- calculate space filling subset ----------------------------------
            design = StandardScaler().fit_transform(design)
            prelim_subset = psa_select(design, n_points, selection_target='max_dist_from_boundary')

            # return training data for outputs and corresponding inputs
            index = np.vstack(
                np.array(
                    np.all((design[:, None, :] == prelim_subset[None, :, :]), axis=-1).nonzero()
                ).T.tolist()
            )[:, 0]
            self.X_train = self.model.X_mc[index, :]
            self.Y_LFs_train = self.model.Y_LFs_mc[index, :]

        else:
            raise NotImplementedError(
                f"You specified the non-valid method"
                f" '{self.initial_design['method']}'. This is not "
                f"implemented! The only valid methods are 'random' or "
                f"'diverse_subset'. Abort..."
            )

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

    def active_learning(self):
        """
        Not implemented yet
        """
        pass

    # ------------------- BELOW JUST PLOTTING AND SAVING RESULTS ------------------
    def post_run(self):
        """
        Saving and plotting of the results. The latter will be moved to a separate module in the
        future.

        Returns:
            None
        """

        if self.result_description['plot_results'] is True:
            plt.rcParams["mathtext.fontset"] = "cm"
            plt.rcParams.update({'font.size': 28})
            fig, ax = plt.subplots()

            min_x = np.min(self.model.y_pdf_support)
            max_x = np.max(self.model.y_pdf_support)
            min_y = 0
            max_y = 1.1 * max(self.output['p_yhf_mc'])
            #            ax.set(xlim=(min_x, max_x), ylim=(min_y, max_y))
            ax.set(xlim=(-0.5, 2), ylim=(min_y, max_y))
            #  --------------------- PLOT THE BMFMC POSTERIOR PDF VARIANCE ---------------------
            if self.predictive_var is True:
                # ax.plot(self.output['y_pdf_support'],self.output['p_yhf_mean']
                #        +np.sqrt(self.output['p_yhf_var']),
                #        linewidth=0.8, color='green', alpha=0.8,
                #        label=r'$\mathbb{SD}\left[\mathrm{p}\left(y_{\mathrm{HF}}|
                #        f^*,\mathcal{D}\right)\right]$')
                # ax.plot(self.output['y_pdf_support'],self.output['p_yhf_mean']-
                # np.sqrt(self.output['p_yhf_var']),
                #        linewidth=0.8, color='green', alpha=0.8,
                #        label=r'$\mathbb{SD}
                #        \left[\mathrm{p}\left(y_{\mathrm{HF}}|f^*,\mathcal{D}\right)\right]$')
                ub = self.output['p_yhf_mean'] + 2 * np.sqrt(self.output['p_yhf_var'])
                lb = self.output['p_yhf_mean'] - 2 * np.sqrt(self.output['p_yhf_var'])
                ax.fill_between(
                    self.output['y_pdf_support'],
                    ub,
                    lb,
                    where=ub > lb,
                    facecolor='lightgrey',
                    alpha=0.5,
                    interpolate=True,
                    label=r'$\pm2\cdot\mathbb{SD}_{f^*}\left[p\left(y_{\mathrm{HF}}^*'
                    r'|f^*,\mathcal{D}_f\right)\right]$',
                )

            # --------------------- PLOT THE BMFMC POSTERIOR PDF MEAN ---------------------
            ax.plot(
                self.output['y_pdf_support'],
                self.output['p_yhf_mean'],
                color='xkcd:green',
                linewidth=3,
                label=r'$\mathrm{\mathbb{E}}_{f^*}\left[p\left(y^*_{\mathrm{HF}}|'
                r'f^*,\mathcal{D}_f\right)\right]$',
            )

            # --------------- PLOT THE REFERENCE SOLUTION FOR CLASSIC BMFMC ---------------
            if self.BMFMC_reference is True:
                # plot the bmfmc var
                if self.predictive_var is True:
                    # ax.plot(self.output['y_pdf_support'], self.output['p_yhf_mean_BMFMC']+
                    # np.sqrt(self.output['p_yhf_var_BMFMC']),linewidth=0.8, color='magenta',
                    # alpha=0.8,label=r'$\mathbb{SD}\left[\mathrm{p}
                    # \left(y_{\mathrm{HF}}|f^*,\mathcal{D}\right)\right]$')
                    # ax.plot(self.output['y_pdf_support'], self.output['p_yhf_mean_BMFMC']-
                    # np.sqrt(self.output['p_yhf_var_BMFMC']),linewidth=0.8, color='magenta',
                    # alpha=0.8,label=r'$\mathbb{SD}\left[\mathrm{p}
                    # \left(y_{\mathrm{HF}}|f^*,\mathcal{D}\right)\right]$')
                    #      ax.plot(self.output['y_pdf_support'],
                    #      np.sqrt(self.output['p_yhf_var_BMFMC']),linewidth=0.8, color='magenta',
                    #      alpha=0.8,label=r'$\mathbb{SD}\left[\mathrm{p}
                    #      \left(y_{\mathrm{HF}}|f^*,\mathcal{D}\right)\right]$')
                    ub = self.output['p_yhf_mean_BMFMC'] + 2 * np.sqrt(
                        self.output['p_yhf_var_BMFMC']
                    )
                    lb = self.output['p_yhf_mean_BMFMC'] - 2 * np.sqrt(
                        self.output['p_yhf_var_BMFMC']
                    )
                    ax.fill_between(
                        self.output['y_pdf_support'],
                        ub,
                        lb,
                        where=ub > lb,
                        facecolor='lightgrey',
                        alpha=0.5,
                        interpolate=True,
                    )  # label=r'$\pm2\cdot\mathbb{SD}_{f^*}\left[p\left(y_{\mathrm{HF}}^*
                    # |f^*,\mathcal{D}_f\right)\right]$')

                # plot the bmfmc approx mean
                ax.plot(
                    self.output['y_pdf_support'],
                    self.output['p_yhf_mean_BMFMC'],
                    color='xkcd:green',
                    linewidth=1.5,
                    linestyle='--',
                    alpha=1,
                    label=r'$\mathrm{\mathbb{E}}_{f^*}\left[p\left(y^*_{\mathrm{HF}}'
                    r'|f^*,\mathcal{D}_f\right)\right],\ (\mathrm{no\ features})$',
                )

            # ------------------------ PLOT THE MC REFERENCE OF HF ------------------------
            ax.plot(
                self.model.y_pdf_support,
                self.output['p_yhf_mc'],
                color='black',
                linestyle='-.',
                linewidth=3,
                alpha=1,
                label=r'$p\left(y_{\mathrm{HF}}\right),\ (\mathrm{MC-ref.})$',
            )
            # plot the MC of LF
            if self.model.Y_LFs_mc.shape[1] < 2:
                ax.plot(
                    self.model.y_pdf_support,
                    self.output['p_ylf_mc'],
                    linewidth=1.5,
                    color='r',
                    alpha=0.8,
                    label=r'$p\left(y_{\mathrm{LF}}\right)$',
                )

            ax.set_xlabel(r'$y$')  # ,usetex=True)
            ax.set_ylabel(r'$p(y)$')
            ax.grid(which='major', linestyle='-')
            ax.grid(which='minor', linestyle='--', alpha=0.5)
            ax.minorticks_on()
            ax.legend(loc='upper right')
            ax.set_xlim(-0.5, 2.0)  # DG: 0.02,0.08
            ax.set_ylim(0, 1.6)  # max DG: 150
            fig.set_size_inches(15, 15)
            # plt.savefig('/home/nitzler/Documents/Vorlagen/pdfs_cylinder_50_nof_random.eps',
            # format='eps', dpi=300)

            # ---------------------- OPTIONAL PLOT OF LATENT MANIFOLD ---------------------
            if self.output['Z_mc'].shape[1] < 2:
                fig2, ax2 = plt.subplots()
                ax2.plot(
                    self.model.Y_LFs_mc[:, 0],
                    self.model.Y_HF_mc,
                    linestyle='',
                    markersize=5,
                    marker='.',
                    color='grey',
                    alpha=0.5,
                    label=r'$\mathcal{D}_{\mathrm{ref}}='
                    r'\{Y_{\mathrm{LF}}^*,Y_{\mathrm{HF}}^*\}$, (Reference)',
                )

                ax2.plot(
                    np.sort(self.output['Z_mc'][:, 0]),
                    self.output['m_f_mc'][np.argsort(self.output['Z_mc'][:, 0])],
                    color='darkblue',
                    linewidth=3,
                    label=r'$\mathrm{m}_{\mathcal{D}_f}(y_{\mathrm{LF}})$, (Posterior mean)',
                )

                ax2.plot(
                    np.sort(self.output['Z_mc'][:, 0]),
                    np.add(self.output['m_f_mc'], np.sqrt(self.output['var_f_mc']))[
                        np.argsort(self.output['Z_mc'][:, 0])
                    ],
                    color='darkblue',
                    linewidth=2,
                    linestyle='--',
                    label=r'$\mathrm{m}_{\mathcal{D}_f}(y_{\mathrm{LF}})\pm \sqrt{\mathrm{v}_'
                    r'{\mathcal{D}_f}(y_{\mathrm{LF}})}$, (Confidence)',
                )

                ax2.plot(
                    np.sort(self.output['Z_mc'][:, 0]),
                    np.add(self.output['m_f_mc'], -np.sqrt(self.output['var_f_mc']))[
                        np.argsort(self.output['Z_mc'][:, 0])
                    ],
                    color='darkblue',
                    linewidth=2,
                    linestyle='--',
                )

                #  ax2.plot(self.output['Z_mc'][:, 0],
                #  self.output['sample_mat'], linestyle='',
                #          marker='.', color='red', alpha=0.2)
                ax2.plot(
                    self.Y_LFs_train,
                    self.model.Y_HF_train,
                    linestyle='',
                    marker='x',
                    markersize=8,
                    color='r',
                    alpha=1,
                    label=r'$\mathcal{D}_{f}=\{Y_{\mathrm{LF}},Y_{\mathrm{HF}}\}$, (Training)',
                )

                ax2.plot(
                    self.model.Y_HF_mc,
                    self.model.Y_HF_mc,
                    linestyle='-',
                    marker='',
                    color='g',
                    alpha=1,
                    linewidth=3,
                    label=r'$y_{\mathrm{HF}}=y_{\mathrm{LF}}$, (Identity)',
                )

                ax2.set_xlabel(r'$y_{\mathrm{LF}}$')  # ,usetex=True)
                ax2.set_ylabel(r'$y_{\mathrm{HF}}$')
                ax2.grid(which='major', linestyle='-')
                ax2.grid(which='minor', linestyle='--', alpha=0.5)
                ax2.minorticks_on()
                ax2.legend()
                fig2.set_size_inches(15, 15)
            # plt.savefig('/home/nitzler/Documents/Vorlagen/ylf_yhf_LF2.eps', format='png', dpi=300)

            if self.output['Z_mc'].shape[1] == 2:
                fig3 = plt.figure(figsize=(10, 10))
                ax3 = fig3.add_subplot(111, projection='3d')

                #  # Normalization of output quantities
                #  self.output['Z_mc'][:, 0, None] =
                #  (self.output['Z_mc'][:, 0, None]
                #               - min(self.output['Z_mc'][:, 0, None])) /\
                #              (max(self.output['Z_mc'][:, 0, None])
                #               - min(self.output['Z_mc'][:, 0, None]))
                #
                #     self.output['Z_mc'][:, 1, None] =
                #     (self.output['Z_mc'][:, 1, None]
                #          - min(self.output['Z_mc'][:, 1, None])) /\
                #         (max(self.output['Z_mc'][:, 1, None])
                #          - min(self.output['Z_mc'][:, 1, None]))\
                #
                #    self.Y_HF_mc[:, None] = (self.Y_HF_mc[:, None] - min(self.Y_HF_mc[:, None])) /\
                #                          (max(self.Y_HF_mc[:, None]) - min(self.Y_HF_mc[:, None]))
                ax3.plot_trisurf(
                    self.output['Z_mc'][:, 0],
                    self.output['Z_mc'][:, 1],
                    self.output['m_f_mc'][:, 0],
                    shade=True,
                    cmap='jet',
                    alpha=0.50,
                )  # , label='$\mathrm{m}_{\mathbf{f}^*}$')
                ax3.scatter(
                    self.output['Z_mc'][:, 0, None],
                    self.output['Z_mc'][:, 1, None],
                    self.model.Y_HF_mc[:, None],
                    s=4,
                    alpha=0.7,
                    c='k',
                    linewidth=0.5,
                    cmap='jet',
                    label='$\mathcal{D}_{\mathrm{MC}}$, (Reference)',
                )

                #  ax3.scatter(self.output['Z_mc'][:, 0, None]*0,
                #  self.output['Z_mc'][:, 1, None],
                #              self.Y_HF_mc[:, None], s=10, alpha=0.05,c=self.Y_HF_mc, cmap='jet')
                #  ax3.scatter(self.output['Z_mc'][:, 0, None],
                #  self.output['Z_mc'][:, 1, None]*0,
                #              self.Y_HF_mc[:, None], s=10, alpha=0.05,c=self.Y_HF_mc, cmap='jet')
                #  ax3.scatter(self.output['Z_mc'][:, 0, None],
                #  self.output['Z_mc'][:, 1, None],
                #              self.Y_HF_mc[:, None]*0, s=10, alpha=0.05,c=self.Y_HF_mc, cmap='jet')

                ax3.scatter(
                    self.output['Z_train'][:, 0, None],
                    self.output['Z_train'][:, 1, None],
                    self.model.Y_HF_train[:, None],
                    marker='x',
                    s=70,
                    c='r',
                    alpha=1,
                    label='$\mathcal{D}$, (Training)',
                )
                # ax3.scatter(self.output['Z_train'][:, 0, None],
                # self.output['Z_train'][:,1,None],
                #             self.Y_HF_train[:, None]*0, marker='x',s=70, c='r', alpha=1)
                # ax3.scatter(self.output['Z_mc'][:, 0, None],
                # self.output['Z_mc'][:,1,None],
                #             self.output['m_f_mc'][:, None], s=10, c='red', alpha=1)

                ax3.set_xlabel(r'$\mathrm{y}_{\mathrm{LF}}$')  # ,usetex=True)
                ax3.set_ylabel(r'$\gamma$')
                ax3.set_zlabel(r'$\mathrm{y}_{\mathrm{HF}}$')

                minx = np.min(self.output['Z_mc'])
                maxx = np.max(self.output['Z_mc'])
                ax3.set_xlim3d(minx, maxx)
                ax3.set_ylim3d(minx, maxx)
                ax3.set_zlim3d(minx, maxx)

                ax3.set_xticks(np.arange(0, 0.5, step=0.5))
                ax3.set_yticks(np.arange(0, 0.5, step=0.5))
                ax3.set_zticks(np.arange(0, 0.5, step=0.5))
                ax3.legend()

            #  # Animate
            #  def init():
            #      ax3.scatter(self.output['Z_mc'][:,0,None],
            #      self.output['Z_mc'][:,1,None],self.Y_HF_mc[:,None],s=3,c='darkgreen',
            #      alpha=0.6)
            #      ax3.set_xlabel(r'$y_{\mathrm{LF}}$')#,usetex=True)
            #      ax3.set_ylabel(r'$\gamma$')
            #      ax3.set_zlabel(r'$y_{\mathrm{HF}}$')
            #      ax3.set_xlim3d(0, 1)
            #      ax3.set_ylim3d(0, 1)
            #      ax3.set_zlim3d(0, 1)
            #
            #      return ()
            #
            #  def animate(i):
            #      ax3.view_init(elev=10., azim=i)
            #      return ()
            #
            #  anim = animation.FuncAnimation(fig3, animate, init_func=init,
            #                                 frames=360, interval=20, blit=True)
            #  # Save
            #  anim.save('cylinder_split_input.mp4', fps=30, dpi=300,
            #  extra_args=['-vcodec', 'libx264'])
            #
            plt.show()

        if self.result_description['write_results'] is True:
            pass
        # ----------------------- CALCULATE SOME KL DIVERGENCES -----------------------
        from scipy.stats import entropy as ep


#  entropy = ep(self.output['p_yhf_mc'],self.output['p_yhf_mean'])
#  entropy_mc = ep(self.output['p_yhf_mc'],self.output['p_yhf_LF_mc'])
#  # append a file
#  with open('cylinder_KLD_new50.txt','a') as myfile:
#      myfile.write('%s\n' % entropy)
#
#  import pandas as pd
#  entropy0 = (pd.read_csv('entropy_random0_fsi.txt', sep=' ',header=None).iloc[:].to_numpy())
#  entropy1 = (pd.read_csv('entropy_random1_fsi.txt', sep=' ',header=None).iloc[:].to_numpy())
#  entropy2 = (pd.read_csv('entropy_random2_fsi.txt', sep=' ',header=None).iloc[:].to_numpy())
#  entropy3 = (pd.read_csv('entropy_random3_fsi.txt', sep=' ',header=None).iloc[:].to_numpy())
#  entropy_fill2 = (pd.read_csv('cylinder_KLD_new50.txt', sep=' ',
#  header=None).iloc[:].to_numpy()[:])
#  entropy_fill = (pd.read_csv('cylinder_KLD_new.txt', sep=' ',header=None).iloc[:].to_numpy()[:])
#  int_var = (pd.read_csv('cylinder_int_var.txt', sep=' ',header=None).iloc[:].to_numpy())
#
#  entropy = (np.hstack((entropy0,entropy1,entropy2, entropy3)))
#
#  mean = np.mean(entropy,axis=0)
#  error = np.std(entropy,axis=0)
#
#  plt.rcParams["mathtext.fontset"] = "cm"
#  plt.rcParams.update({'font.size':28})
#
#  fig,ax1 = plt.subplots()
#  import pandas as pd
#  count = np.arange(0,4)
#  count2 = np.arange(0,6)
#  width = 0.25

#  full=ax1.bar(count2-width/2,entropy_fill[:,0],width, alpha=0.5,color='slategrey',
#  label='Diverse subset, $(\mathbf{n}=50)$',edgecolor='white')
#  data = [entropy0[:,0], entropy1[:,0], entropy2[:,0], entropy3[:,0]]
#  boxes = ax1.boxplot(data, positions=count-width, manage_ticks=False,
#  patch_artist=True, widths=(0.25, 0.25, 0.25, 0.25), showfliers=False )
#  for box in boxes['boxes']:
#      box.set( facecolor=r'blue')


# ax1.bar(count,mean,-width,yerr=error,alpha=0.5,ecolor='black',capsize=10, color='b',
# label='Random in bins, ($\mathbf{n}=200$)',edgecolor='white')
#        semi_full=ax1.bar(count2+width/2,entropy_fill2[:,0],width, alpha=0.5,color='seagreen',
#        label='Diverse subset, $(\mathbf{n}=30)$',edgecolor='white')
#        ax1.set_xlabel(r'Number of features')
#        ax1.set_ylabel(r'$\mathrm{D}_{\mathrm{KL}}\left[p\left(y_{\mathrm{HF}}\right)||
#        \mathrm{\mathbb{E}}_{f^*}
#        \left[p\left(y_{\mathrm{HF}}^*|f^*,\mathcal{D}_f\right)\right]\right]$')
#        ax1.xaxis.set_ticks(np.arange(0, 6, 1))
##
#        ax1.grid(which='major', linestyle='-')
#        ax1.grid(which='minor', linestyle='--', alpha=0.5)
#        ax1.minorticks_on()
#
#        mc_50 = 0.5948700915902669
#        mc_150 = 0.3628080477634491
#        mc_500 = 0.14934914831115265
#        mc_1000 = 0.020575027401092127
#        mc_5000 = 0.00944536197510016
#
#        ax1.plot([count2[0]-width, count2[-1]+width], [mc_50, mc_50], "k--", label='mc 50')
#        ax1.plot([count2[0]-width, count2[-1]+width], [mc_150, mc_150], "k--", label='mc 150')
#        ax1.plot([count2[0]-width, count2[-1]+width], [mc_500, mc_500], "k--", label='mc 500')
#        ax1.plot([count2[0]-width, count2[-1]+width], [mc_1000, mc_1000], "k--", label='mc 1000')
#        ax1.plot([count2[0]-width, count2[-1]+width], [mc_5000, mc_5000], "k--", label='mc 5000')
#        print(entropy_mc)
#        ax1.set_yscale('log')
#        ax1.set_ylim((0,1.0))
#        ax1.legend([full[0],semi_full[0]],['Diverse subset, $\mathrm{n}_{\mathrm{train}}=150$',
#        'Diverse subset, $\mathrm{n}_{\mathrm{train}}=50$'],loc='upper left')
#
#        fig.set_size_inches(15, 15)
#        plt.savefig('/home/nitzler/Documents/Vorlagen/kld_reduction_cylinder.eps',
#        format='eps', dpi=300)

#   integ=ax1.bar(count2,int_var[:,0],width, alpha=0.5,color='seagreen',
#   label='Diverse subset, $(\mathbf{n}=30)$',edgecolor='white')
#    ax1.set_xlabel(r'Number of features')
#     ax1.set_ylabel(r'$IV$')
#      ax1.set_ylim((0,0.0015))
#       ax1.xaxis.set_ticks(np.arange(0, 6, 1))
#
#
#       ax1.grid(which='major', linestyle='-')
#       ax1.grid(which='minor', linestyle='--', alpha=0.5)
#        ax1.minorticks_on()
#        import matplotlib.ticker as mtick
#       ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
#       fig.set_size_inches(15, 15)
#       plt.savefig('/home/nitzler/Documents/Vorlagen/IV_cylinder.eps', format='eps', dpi=300)
#        plt.show()
