import numpy as np
from pqueens.models.bmfmc_model import BMFMCModel
from scipy.spatial.distance import pdist, squareform
import pandas as pd
from .iterator import Iterator
from pqueens.iterators.data_iterator import DataIterator
import os
from random import randint
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.preprocessing import StandardScaler
from diversipy import *


class BmfmcIterator(Iterator):
    """ Basic BMFMC Iterator to enable selective sampling for the hifi-lofi
        model (the basis was a standard LHCIterator). Additionally the BMFMC
        Iterator contains the subiterator data_iterator to make use of already
        performed MC samples on the lo-fi models

    Attributes:
        model (model):        multi-fidelity model comprising sub-models
        num_samples (int):    Number of samples to compute
        path_to_lofi_mc_data:
        name_of_mc_model:
        path_to_point_list:
        result_description (dict):  Description of desired results (write data)
        samples (np.array):   Array with all samples
        outputs (np.array):   Array with all model outputs

    """

    def __init__(
        self,
        config,
        lf_data_iterators,
        hf_data_iterator,
        result_description,
        experiment_dir,
        initial_design,
        predictive_var,
        BMFMC_reference,
        global_settings,
    ):

        super(BmfmcIterator, self).__init__(
            None, global_settings
        )  # Input prescribed by iterator.py
        self.model = None
        self.result_description = result_description
        self.lf_data_iterators = lf_data_iterators
        self.hf_data_iterator = hf_data_iterator
        self.experiment_dir = experiment_dir
        self.train_in = None
        self.hf_train_out = None
        self.lfs_train_out = None
        self.lf_mc_in = None
        self.hf_mc = None
        self.lfs_mc_out = None
        self.eigenfunc = None
        self.output = None  # this is still important and will prob be the marginal pdf of the hf / its statistics
        self.initial_design = initial_design
        self.predictive_var = predictive_var
        self.config = config  # This is needed to construct the model on runtime
        self.BMFMC_reference = BMFMC_reference

    @classmethod
    def from_config_create_iterator(cls, config, iterator_name=None):
        """ Create LHS iterator from problem description

        Args:
            config (dict): Dictionary with QUEENS problem description
            iterator_name (str): Name of iterator to identify right section in options dict
            model (model): model to use

        Returns:
            iterator: BmfmcIterator object

        """
        # Initialize Iterator and model
        method_options = config["method"]["method_options"]
        BMFMC_reference = method_options["BMFMC_reference"]
        result_description = method_options["result_description"]
        experiment_dir = method_options["experiment_dir"]
        predictive_var = method_options["predictive_var"]

        # Create the data iterator from config file
        lf_data_paths = method_options[
            "path_to_lf_data"
        ]  # necessary to substract that name from name list of all models for evaluation
        hf_data_path = method_options["path_to_hf_data"]
        lf_data_iterators = []
        for _, path in enumerate(lf_data_paths):
            lf_data_iterators.append(DataIterator(path, None, None))
        hf_data_iterator = DataIterator(hf_data_path, None, None)
        initial_design = config["method"]["initial_design"]
        global_settings = config.get('global_settings', None)

        return cls(
            config,
            lf_data_iterators,
            hf_data_iterator,
            result_description,
            experiment_dir,
            initial_design,
            predictive_var,
            BMFMC_reference,
            global_settings,
        )

    def core_run(self):
        """
        Generate simulation runs for LF and HF that cover the output
        space based on one p(y) distribution of one lofi
        """
        # ------------------------------ LOAD LF MC DATA ------------------------------
        self.lf_mc_in = self.lf_data_iterators[0].read_pickle_file()[
            0
        ]  # here we assume that all lfs have the same input vector
        try:
            self.eigenfunc = self.lf_data_iterators[0].read_pickle_file()[-1]
        except IOError:
            self.eigenfunc = None
        lfs_mc_out = [
            lf_data_iterator.read_pickle_file()[1][:, 0]
            for _, lf_data_iterator in enumerate(self.lf_data_iterators)
        ]
        self.lfs_mc_out = np.atleast_2d(np.vstack(lfs_mc_out)).T

        #  CREATE HF DATA --> CURRENTLY WE SELECT FROM HF MC DATA SET AND DO NOT TRIGGER ACTUAL SIMUALTIONS HERE
        # -----------------  HENCE METHOD BELOW IS CURRENTLY NOT USED -----------------
        if self.hf_data_iterator is None:  #  TODO: This needs still to be done and is a placeholder
            minval = mc_y.min()
            maxval = mc_y.max()
            n_bins = self.num_samples // 2
            break_points = np.linspace(minval, maxval, n_bins + 1)

            # Binning using cut function of pandas -> gives every y a bin number
            bin_vec = pd.cut(
                mc_y, bins=break_points, labels=False, include_lowest=True, retbins=True
            )

            # calculate most distant point pair per bin
            mapping_points_x = []
            mapping_points_y = []
            for bin_n in range(self.num_samples // 2):
                # create boolean vector with length of y that filters all y belonging
                # to one active bin
                boolean_vec = [
                    bin_vec == bin_n
                ]  # Check if this is enough to create a boolean vec or if for loop is necessary
                bin_data_x = mc_x[boolean_vec]
                bin_data_y = mc_y[boolean_vec]

                # (furtherst neighbor algorithm), design of data: y,x1,x2,x3....
                D = pdist(bin_data_x)
                D = squareform(D)
                [row, col] = np.unravel_index(np.argmax(D), D.shape)
                mapping_points_x.append(bin_data_x[row, col])
                mapping_points_y.append(bin_data_y[row, col])

            # self.samples = mapping_points_x
            self.mc_map_output = mapping_points_y
        else:
            #   THIS IS THE APPROACH CURRENTLY USED -> SELECT HF TRAINING DATA FROM MC DATA -> CAREFUL: WORKAROUND!
            # check if training data file already exists if not write it
            path_to_train_data = os.path.join(self.experiment_dir, 'hf_lf_train.pickle')
            exists = os.path.isfile(path_to_train_data)
            exists = None  # TODO delete! this is just to not write the file
            if exists:
                with open(
                    path_to_train_data, 'rb'
                ) as handle:  # TODO: give better name according to experiment and choose better location
                    # pylint: disable=line-too-long
                    (
                        self.train_in,
                        self.lfs_train_out,
                        self.hf_train_out,
                        self.lf_mc_in,
                        self.lfs_mc_out,
                    ) = pickle.load(
                        handle
                    )  # TODO --> Change this according to design below!!# pylint: enable=line-too-long
            else:
                self.train_in, self.hf_train_out, _ = self.hf_data_iterator.read_pickle_file()
                self.hf_train_out = self.hf_train_out[
                    :, 0
                ]  # TODO here we neglect the vectorial output --> this should be changed in the future
                self.hf_mc = self.hf_train_out

                # -------------------------- RANDOM FROM BINS METHOD --------------------------
                if self.initial_design['method'] == 'random':
                    n_bins = self.initial_design["num_bins"]
                    n_points = self.initial_design["num_HF_eval"]
                    ylf_min = np.amin(self.lfs_mc_out)
                    ylf_max = np.amax(self.lfs_mc_out)
                    break_points = np.linspace(ylf_min, ylf_max, n_bins + 1)
                    bin_vec = pd.cut(
                        self.lfs_mc_out[:, 0],
                        bins=break_points,
                        labels=False,
                        include_lowest=True,
                        retbins=True,
                    )  # TODO check dim of lfs_mc_out for multiple lfs
                    # Some initialization
                    self.lfs_train_out = np.empty(
                        (0, self.lfs_mc_out.shape[1])
                    )  # TODO check if this is right
                    dummy_hf = self.hf_train_out  # TODO delete later
                    self.hf_train_out = np.array([]).reshape(
                        0, 1
                    )  # TODO: Later delete as HF MC is normally not available
                    self.train_in = np.array([]).reshape(0, self.lf_mc_in.shape[1])
                    if self.eigenfunc is not None:
                        self.eigenfunc_train = np.array([]).reshape(0, self.eigenfunc.shape[1])

                    # Go through all bins and select randomly select points from them
                    # (also several per bin!)
                    for bin_n in range(n_bins):
                        # array of booleans
                        y_in_bin_bool = [bin_vec[0] == bin_n]
                        # definine bin data
                        bin_data_in = self.lf_mc_in[tuple(y_in_bin_bool)]
                        bin_data_lfs = self.lfs_mc_out[tuple(y_in_bin_bool)]
                        bin_data_hf = dummy_hf[tuple(y_in_bin_bool)].reshape(
                            -1, 1
                        )  # TODO: later delete as HF MC is normally not available
                        # randomly select points in bins
                        rnd_select = [
                            randint(0, bin_data_lfs.shape[0] - 1) for p in range(n_points // n_bins)
                        ]
                        # Check if points in bin
                        if len(rnd_select) != 0:
                            # define the training data by appending the training vector
                            self.train_in = np.vstack([self.train_in, bin_data_in[rnd_select, :]])
                            self.lfs_train_out = np.vstack(
                                (self.lfs_train_out, bin_data_lfs[rnd_select, :])
                            )
                            self.hf_train_out = np.vstack(
                                [self.hf_train_out, bin_data_hf[rnd_select, :]]
                            )  # TODO later delete as HF MC data is normally not available

                # --------------------------- DIVERSE SUBSET METHOD ---------------------------
                elif self.initial_design['method'] == 'diverse_subset':
                    n_points = self.initial_design["num_HF_eval"]
                    # random_fields_test = self.lf_mc_in[:, 3:] # DG
                    random_fields_test = self.lf_mc_in[:, 1:]  # FSI
                    x_vec = np.linspace(0, 1, 200, endpoint=True)
                    mean_fun = 4 * 1.5 * (-((x_vec - 0.5) ** 2) + 0.25)
                    normalized_test = random_fields_test  # FSI

                    # coef_test = np.dot(self.eigenfunc.T, normalized_test.T).T # DG
                    # design =  np.hstack((self.lf_mc_in[:,0:3],coef_test[:,0:5])) # self.lfs_mc_out[:])) # DG
                    design = np.hstack((self.lf_mc_in[:, 0:1], self.lfs_mc_out[:]))  # FSI

                    design = StandardScaler().fit_transform(design)
                    prelim_subset = psa_select(
                        design, n_points, selection_target='max_dist_from_boundary'
                    )  # calculate space filling subset

                    # return training data for outputs and corresponding inputs
                    index = np.vstack(
                        np.array(
                            np.all(
                                (design[:, None, :] == prelim_subset[None, :, :]), axis=-1
                            ).nonzero()
                        ).T.tolist()
                    )[:, 0]
                    self.train_in = self.lf_mc_in[index, :]
                    self.lfs_train_out = self.lfs_mc_out[index, :]
                    self.hf_train_out = self.hf_mc[
                        index, None
                    ]  # TODO this is an intermediate solution cause we already have the dataset

                else:
                    pass  # TODO this needs to be changed to an keyword error check

                # ------ WRITE NEW PICKLE FILE FOR SUBSEQUENT ANALYSIS WITH MATCHING DATA -----
                with open(
                    path_to_train_data, 'wb'
                ) as handle:  # TODO: give better name according to experiment and choose better location
                    pickle.dump(
                        [
                            self.train_in,
                            self.lfs_train_out,
                            self.hf_train_out,
                            self.lf_mc_in,
                            self.lfs_mc_out,
                        ],
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )

        # ----------------------------- CREATE BMFMC MODEL ----------------------------
        self.model = BMFMCModel.from_config_create_model(
            self.config,
            self.train_in,
            self.lfs_train_out,
            self.hf_train_out,
            self.lf_mc_in,
            self.lfs_mc_out,
            hf_mc=self.hf_mc,
            eigenfunc=self.eigenfunc,
        )

        # TODO: This needs still to be implemented to run HF directly from this iterator
        #  for initial desing
        """ Run Analysis on all models """
        # here the set of new input variables will be passed to all lofis/hifi -->
        # previous list_iterator not necessary!
        # self.model.update_model_from_sample_batch(self.samples)

        # here all models will be evaluated (multifidelity  model will be evaluated)
        # output matrix has the form:[ ylofi1, ylofi2, ..., yhifi ]

        self.output = self.eval_model()
        #  TODO some active learning here

    def eval_model(self):
        """ Evaluate the model """
        return self.model.evaluate()

    def active_learning(self):
        pass

    # ------------------- BELOW JUST PLOTTING AND SAVING RESULTS ------------------
    def post_run(self):
        """ Analyze the results """

        if self.result_description['plot_results'] is True:
            plt.rcParams["mathtext.fontset"] = "cm"
            plt.rcParams.update({'font.size': 28})
            fig, ax = plt.subplots()

            min_x = 0.5 * min(
                [min(self.output['pyhf_support']), min(self.output['pylf_mc_support'])]
            )
            max_x = (
                0.1  # max([max(self.output['pyhf_support']), max(self.output['pylf_mc_support'])])
            )
            min_y = 0
            max_y = 1.1 * max(self.output['pyhf_mc'])
            #            ax.set(xlim=(min_x, max_x), ylim=(min_y, max_y))
            ax.set(xlim=(-0.5, 2), ylim=(min_y, max_y))

            #  --------------------- PLOT THE BMFMC POSTERIOR PDF VARIANCE ---------------------
            if self.predictive_var is True:
                # ax.plot(self.output['pyhf_support'],self.output['pyhf_mean']+np.sqrt(self.output['pyhf_var']),
                #        linewidth=0.8, color='green', alpha=0.8, label=r'$\mathbb{SD}\left[\mathrm{p}\left(y_{\mathrm{HF}}|f^*,\mathcal{D}\right)\right]$')
                # ax.plot(self.output['pyhf_support'],self.output['pyhf_mean']-np.sqrt(self.output['pyhf_var']),
                #        linewidth=0.8, color='green', alpha=0.8, label=r'$\mathbb{SD}\left[\mathrm{p}\left(y_{\mathrm{HF}}|f^*,\mathcal{D}\right)\right]$')
                ub = self.output['pyhf_mean'] + 2 * np.sqrt(self.output['pyhf_var'])
                lb = self.output['pyhf_mean'] - 2 * np.sqrt(self.output['pyhf_var'])
                ax.fill_between(
                    self.output['pyhf_support'],
                    ub,
                    lb,
                    where=ub > lb,
                    facecolor='lightgrey',
                    alpha=0.5,
                    interpolate=True,
                    label=r'$\pm2\cdot\mathbb{SD}_{f^*}\left[p\left(y_{\mathrm{HF}}^*|f^*,\mathcal{D}_f\right)\right]$',
                )

            # --------------------- PLOT THE BMFMC POSTERIOR PDF MEAN ---------------------
            ax.plot(
                self.output['pyhf_support'],
                self.output['pyhf_mean'],
                color='xkcd:green',
                linewidth=3,
                label=r'$\mathrm{\mathbb{E}}_{f^*}\left[p\left(y^*_{\mathrm{HF}}|f^*,\mathcal{D}_f\right)\right]$',
            )

            # --------------- PLOT THE REFERENCE SOLUTION FOR CLASSIC BMFMC ---------------
            if self.BMFMC_reference is True:
                # plot the bmfmc var
                if self.predictive_var is True:
                    # ax.plot(self.output['pyhf_support'], self.output['pyhf_mean_BMFMC']+np.sqrt(self.output['pyhf_var_BMFMC']),linewidth=0.8, color='magenta', alpha=0.8,label=r'$\mathbb{SD}\left[\mathrm{p}\left(y_{\mathrm{HF}}|f^*,\mathcal{D}\right)\right]$')
                    # ax.plot(self.output['pyhf_support'], self.output['pyhf_mean_BMFMC']-np.sqrt(self.output['pyhf_var_BMFMC']),linewidth=0.8, color='magenta', alpha=0.8,label=r'$\mathbb{SD}\left[\mathrm{p}\left(y_{\mathrm{HF}}|f^*,\mathcal{D}\right)\right]$')
                    #                    ax.plot(self.output['pyhf_support'], np.sqrt(self.output['pyhf_var_BMFMC']),linewidth=0.8, color='magenta', alpha=0.8,label=r'$\mathbb{SD}\left[\mathrm{p}\left(y_{\mathrm{HF}}|f^*,\mathcal{D}\right)\right]$')
                    ub = self.output['pyhf_mean_BMFMC'] + 2 * np.sqrt(self.output['pyhf_var_BMFMC'])
                    lb = self.output['pyhf_mean_BMFMC'] - 2 * np.sqrt(self.output['pyhf_var_BMFMC'])
                    ax.fill_between(
                        self.output['pyhf_support'],
                        ub,
                        lb,
                        where=ub > lb,
                        facecolor='lightgrey',
                        alpha=0.5,
                        interpolate=True,
                    )  # label=r'$\pm2\cdot\mathbb{SD}_{f^*}\left[p\left(y_{\mathrm{HF}}^*|f^*,\mathcal{D}_f\right)\right]$')

                # plot the bmfmc approx mean
                ax.plot(
                    self.output['pyhf_support'],
                    self.output['pyhf_mean_BMFMC'],
                    color='xkcd:green',
                    linewidth=1.5,
                    linestyle='--',
                    alpha=1,
                    label=r'$\mathrm{\mathbb{E}}_{f^*}\left[p\left(y^*_{\mathrm{HF}}|f^*,\mathcal{D}_f\right)\right],\ (\mathrm{no\ features})$',
                )

            # ------------------------ PLOT THE MC REFERENCE OF HF ------------------------
            ax.plot(
                self.output['pyhf_mc_support'],
                self.output['pyhf_mc'],
                color='black',
                linestyle='-.',
                linewidth=3,
                alpha=1,
                label=r'$p\left(y_{\mathrm{HF}}\right),\ (\mathrm{MC-ref.})$',
            )
            # plot the MC of LF
            if self.lfs_mc_out.shape[1] < 2:
                ax.plot(
                    self.output['pylf_mc_support'],
                    self.output['pylf_mc'],
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
            #            plt.savefig('/home/nitzler/Documents/Vorlagen/pdfs_cylinder_50_nof_random.eps', format='eps', dpi=300)

            # ---------------------- OPTIONAL PLOT OF LATENT MANIFOLD ---------------------
            if self.output['manifold_test'].shape[1] < 2:
                fig2, ax2 = plt.subplots()
                ax2.plot(
                    self.lfs_mc_out[:, 0],
                    self.hf_mc,
                    linestyle='',
                    markersize=5,
                    marker='.',
                    color='grey',
                    alpha=0.5,
                    label=r'$\mathcal{D}_{\mathrm{ref}}=\{Y_{\mathrm{LF}}^*,Y_{\mathrm{HF}}^*\}$, (Reference)',
                )

                ax2.plot(
                    np.sort(self.output['manifold_test'][:, 0]),
                    self.output['f_mean'][np.argsort(self.output['manifold_test'][:, 0])],
                    color='darkblue',
                    linewidth=3,
                    label=r'$\mathrm{m}_{\mathcal{D}_f}(y_{\mathrm{LF}})$, (Posterior mean)',
                )

                ax2.plot(
                    np.sort(self.output['manifold_test'][:, 0]),
                    np.add(self.output['f_mean'], np.sqrt(self.output['y_var']))[
                        np.argsort(self.output['manifold_test'][:, 0])
                    ],
                    color='darkblue',
                    linewidth=2,
                    linestyle='--',
                    label=r'$\mathrm{m}_{\mathcal{D}_f}(y_{\mathrm{LF}})\pm \sqrt{\mathrm{v}_{\mathcal{D}_f}(y_{\mathrm{LF}})}$, (Confidence)',
                )

                ax2.plot(
                    np.sort(self.output['manifold_test'][:, 0]),
                    np.add(self.output['f_mean'], -np.sqrt(self.output['y_var']))[
                        np.argsort(self.output['manifold_test'][:, 0])
                    ],
                    color='darkblue',
                    linewidth=2,
                    linestyle='--',
                )

                #                ax2.plot(self.output['manifold_test'][:, 0], self.output['sample_mat'], linestyle='',
                #                         marker='.', color='red', alpha=0.2)
                ax2.plot(
                    self.lfs_train_out,
                    self.hf_train_out,
                    linestyle='',
                    marker='x',
                    markersize=8,
                    color='r',
                    alpha=1,
                    label=r'$\mathcal{D}_{f}=\{Y_{\mathrm{LF}},Y_{\mathrm{HF}}\}$, (Training)',
                )

                ax2.plot(
                    self.hf_mc,
                    self.hf_mc,
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
            #                plt.savefig('/home/nitzler/Documents/Vorlagen/ylf_yhf_LF2.eps', format='png', dpi=300)

            if self.output['manifold_test'].shape[1] == 2:
                fig3 = plt.figure(figsize=(10, 10))
                ax3 = fig3.add_subplot(111, projection='3d')

                #                # Normalization of output quantities
                #                self.output['manifold_test'][:, 0, None] = (self.output['manifold_test'][:, 0, None]
                #                                                            - min(self.output['manifold_test'][:, 0, None])) /\
                #                                                           (max(self.output['manifold_test'][:, 0, None])
                #                                                            - min(self.output['manifold_test'][:, 0, None]))
                #
                #                self.output['manifold_test'][:, 1, None] = (self.output['manifold_test'][:, 1, None]
                #                                                            - min(self.output['manifold_test'][:, 1, None])) /\
                #                                                           (max(self.output['manifold_test'][:, 1, None])
                #                                                            - min(self.output['manifold_test'][:, 1, None]))\
                #
                #                self.hf_mc[:, None] = (self.hf_mc[:, None] - min(self.hf_mc[:, None])) /\
                #                                      (max(self.hf_mc[:, None]) - min(self.hf_mc[:, None]))
                ax3.plot_trisurf(
                    self.output['manifold_test'][:, 0],
                    self.output['manifold_test'][:, 1],
                    self.output['f_mean'][:, 0],
                    shade=True,
                    cmap='jet',
                    alpha=0.50,
                )  # , label='$\mathrm{m}_{\mathbf{f}^*}$')
                ax3.scatter(
                    self.output['manifold_test'][:, 0, None],
                    self.output['manifold_test'][:, 1, None],
                    self.hf_mc[:, None],
                    s=4,
                    alpha=0.7,
                    c='k',
                    linewidth=0.5,
                    cmap='jet',
                    label='$\mathcal{D}_{\mathrm{MC}}$, (Reference)',
                )

                #                ax3.scatter(self.output['manifold_test'][:, 0, None]*0, self.output['manifold_test'][:, 1, None],
                #                            self.hf_mc[:, None], s=10, alpha=0.05,c=self.hf_mc, cmap='jet')
                #                ax3.scatter(self.output['manifold_test'][:, 0, None], self.output['manifold_test'][:, 1, None]*0,
                #                            self.hf_mc[:, None], s=10, alpha=0.05,c=self.hf_mc, cmap='jet')
                #                ax3.scatter(self.output['manifold_test'][:, 0, None], self.output['manifold_test'][:, 1, None],
                #                            self.hf_mc[:, None]*0, s=10, alpha=0.05,c=self.hf_mc, cmap='jet')

                ax3.scatter(
                    self.output['manifold_train'][:, 0, None],
                    self.output['manifold_train'][:, 1, None],
                    self.hf_train_out[:, None],
                    marker='x',
                    s=70,
                    c='r',
                    alpha=1,
                    label='$\mathcal{D}$, (Training)',
                )
                #                ax3.scatter(self.output['manifold_train'][:, 0, None], self.output['manifold_train'][:,1,None],
                #                            self.hf_train_out[:, None]*0, marker='x',s=70, c='r', alpha=1)
                #                ax3.scatter(self.output['manifold_test'][:, 0, None], self.output['manifold_test'][:,1,None],
                #                            self.output['f_mean'][:, None], s=10, c='red', alpha=1)

                ax3.set_xlabel(r'$\mathrm{y}_{\mathrm{LF}}$')  # ,usetex=True)
                ax3.set_ylabel(r'$\gamma$')
                ax3.set_zlabel(r'$\mathrm{y}_{\mathrm{HF}}$')

                minx = np.min(self.output['manifold_test'])
                maxx = np.max(self.output['manifold_test'])
                ax3.set_xlim3d(minx, maxx)
                ax3.set_ylim3d(minx, maxx)
                ax3.set_zlim3d(minx, maxx)

                ax3.set_xticks(np.arange(0, 0.5, step=0.5))
                ax3.set_yticks(np.arange(0, 0.5, step=0.5))
                ax3.set_zticks(np.arange(0, 0.5, step=0.5))
                ax3.legend()

            #                # Animate
            #                def init():
            #                    ax3.scatter(self.output['manifold_test'][:,0,None],self.output['manifold_test'][:,1,None],self.hf_mc[:,None],s=3,c='darkgreen', alpha=0.6)
            #                    ax3.set_xlabel(r'$y_{\mathrm{LF}}$')#,usetex=True)
            #                    ax3.set_ylabel(r'$\gamma$')
            #                    ax3.set_zlabel(r'$y_{\mathrm{HF}}$')
            #                    ax3.set_xlim3d(0, 1)
            #                    ax3.set_ylim3d(0, 1)
            #                    ax3.set_zlim3d(0, 1)
            #
            #                    return ()
            #
            #                def animate(i):
            #                    ax3.view_init(elev=10., azim=i)
            #                    return ()
            #
            #                anim = animation.FuncAnimation(fig3, animate, init_func=init,
            #                                               frames=360, interval=20, blit=True)
            #                # Save
            #                anim.save('cylinder_split_input.mp4', fps=30, dpi=300, extra_args=['-vcodec', 'libx264'])
            #
            plt.show()

        if self.result_description['write_results'] is True:
            pass
        # ----------------------- CALCULATE SOME KL DIVERGENCES -----------------------
        from scipy.stats import entropy as ep


#        entropy = ep(self.output['pyhf_mc'],self.output['pyhf_mean'])
#        entropy_mc = ep(self.output['pyhf_mc'],self.output['pyhf_mc_low'])
#        # append a file
#        with open('cylinder_KLD_new50.txt','a') as myfile:
#            myfile.write('%s\n' % entropy)
#
#        import pandas as pd
#        entropy0 = (pd.read_csv('entropy_random0_fsi.txt', sep=' ',header=None).iloc[:].to_numpy())
#        entropy1 = (pd.read_csv('entropy_random1_fsi.txt', sep=' ',header=None).iloc[:].to_numpy())
#        entropy2 = (pd.read_csv('entropy_random2_fsi.txt', sep=' ',header=None).iloc[:].to_numpy())
#        entropy3 = (pd.read_csv('entropy_random3_fsi.txt', sep=' ',header=None).iloc[:].to_numpy())
#        entropy_fill2 = (pd.read_csv('cylinder_KLD_new50.txt', sep=' ',header=None).iloc[:].to_numpy()[:])
#        entropy_fill = (pd.read_csv('cylinder_KLD_new.txt', sep=' ',header=None).iloc[:].to_numpy()[:])
#        int_var = (pd.read_csv('cylinder_int_var.txt', sep=' ',header=None).iloc[:].to_numpy())
#
#        entropy = (np.hstack((entropy0,entropy1,entropy2, entropy3)))
#
#        mean = np.mean(entropy,axis=0)
#        error = np.std(entropy,axis=0)
#
#        plt.rcParams["mathtext.fontset"] = "cm"
#        plt.rcParams.update({'font.size':28})
#
#        fig,ax1 = plt.subplots()
#        import pandas as pd
#        count = np.arange(0,4)
#        count2 = np.arange(0,6)
#        width = 0.25

#        full=ax1.bar(count2-width/2,entropy_fill[:,0],width, alpha=0.5,color='slategrey', label='Diverse subset, $(\mathbf{n}=50)$',edgecolor='white')
#        data = [entropy0[:,0], entropy1[:,0], entropy2[:,0], entropy3[:,0]]
#        boxes = ax1.boxplot(data, positions=count-width, manage_ticks=False, patch_artist=True, widths=(0.25, 0.25, 0.25, 0.25), showfliers=False )
#        for box in boxes['boxes']:
#            box.set( facecolor=r'blue')


# ax1.bar(count,mean,-width,yerr=error,alpha=0.5,ecolor='black',capsize=10, color='b', label='Random in bins, ($\mathbf{n}=200$)',edgecolor='white')
#        semi_full=ax1.bar(count2+width/2,entropy_fill2[:,0],width, alpha=0.5,color='seagreen', label='Diverse subset, $(\mathbf{n}=30)$',edgecolor='white')
#        ax1.set_xlabel(r'Number of features')
#        ax1.set_ylabel(r'$\mathrm{D}_{\mathrm{KL}}\left[p\left(y_{\mathrm{HF}}\right)||\mathrm{\mathbb{E}}_{f^*}\left[p\left(y_{\mathrm{HF}}^*|f^*,\mathcal{D}_f\right)\right]\right]$')
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
#        ax1.legend([full[0],semi_full[0]],['Diverse subset, $\mathrm{n}_{\mathrm{train}}=150$','Diverse subset, $\mathrm{n}_{\mathrm{train}}=50$'],loc='upper left')
#
#        fig.set_size_inches(15, 15)
#        plt.savefig('/home/nitzler/Documents/Vorlagen/kld_reduction_cylinder.eps', format='eps', dpi=300)

#   integ=ax1.bar(count2,int_var[:,0],width, alpha=0.5,color='seagreen', label='Diverse subset, $(\mathbf{n}=30)$',edgecolor='white')
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
