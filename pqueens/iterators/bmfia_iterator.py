import numpy as np
import numpy.matlib
from pqueens.database.mongodb import MongoDB
from pqueens.external_geometry.external_geometry import ExternalGeometry
from pqueens.iterators.iterator import Iterator
from pqueens.iterators.monte_carlo_iterator import MonteCarloIterator
from pqueens.models.model import Model
from pqueens.utils.process_outputs import process_ouputs, write_results


class BMFIAIterator(Iterator):
    """
    Iterator for Bayesian multi-fidelity inverse analysis. Here, we build the multi-fidelity
    probabilistic surrogate, determine optimal training points X_train and evaluate the low- and
    high-fidelity model for these training inputs to yield Y_LF_train and Y_HF_train.
    training data. The actual inverse problem is not solved or iterated in this module but
    instead we iterate over the training data to approximate the probabilistic mapping p(yhf|ylf).

    Attributes:
        result_description (dict): Dictionary containing settings for result handling and writing
        X_train (np.array): Input training matrix for HF and LF model
        Y_LF_train (np.array): Corresponding LF model response to X_train input
        Y_HF_train (np.array): Corresponding HF model response to X_train input
        Z_train (np.array): Corresponding LF informative features to X_train input
        features_config (str): Type of feature selection method
        hf_model (obj): High-fidelity model object
        lf_model (obj): Low fidelity model object
        coords_experimental_data (np.array): Coordinates of the experimental data
        output_label (str): Name or label of the output quantity of interest (used to find the
                            data in the csv file)
        coord_labels (lst): Label or names of the underlying coordinates for the experimental
                            data. This should be in the same order as the experimental_data array
        y_obs_vec (np.array): Output data of experimental observations
        settings_probab_mapping (dict): Dictionary with settings for the probabilistic
                                        multi-fidelity mapping
        db (obj): Database object
        external_geometry_obj (obj): External geometry object


    Returns:
       BMFIAIterator (obj): Instance of the BMFIAIterator

    """

    def __init__(
        self,
        result_description,
        global_settings,
        features_config,
        hf_model,
        lf_model,
        output_label,
        coord_labels,
        settings_probab_mapping,
        db,
        external_geometry_obj,
        x_train
    ):
        super(BMFIAIterator, self).__init__(
            None, global_settings
        )  # Input prescribed by iterator.py

        self.result_description = result_description
        self.X_train = x_train
        self.Y_LF_train = None
        self.Y_HF_train = None
        self.Z_train = None
        self.features_config = features_config
        self.hf_model = hf_model
        self.lf_model = lf_model
        self.coords_experimental_data = None
        self.output_label = output_label
        self.coord_labels = coord_labels
        self.y_obs_vec = None
        self.settings_probab_mapping = settings_probab_mapping
        self.db = db
        self.external_geometry_obj = external_geometry_obj

    @classmethod
    def from_config_create_iterator(cls, config, iterator_name=None, model=None):
        """
        Build a BMFIAIterator object from the problem description

        Args:
            config (dict): Configuration / input file for QUEENS as dictionary
            iterator_name (str): Iterator name (optional)
            model (str): Model name (optional)

        Returns:
            iterator (obj): BMFIAIterator object

        """
        # Get appropriate sections in the config file
        method_options = config["method"]["method_options"]
        model_name = method_options["model"]
        global_settings = config.get('global_settings', None)
        result_description = method_options["result_description"]

        # get mf approx settings
        mf_approx_settings = config[model_name].get("mf_approx_settings")
        features_config = mf_approx_settings["features_config"]

        # get the mf subiterator settings
        bmfia_iterator_name = mf_approx_settings["mf_subiterator"]
        bmfia_iterator_dict = config[bmfia_iterator_name]
        hf_model_name = bmfia_iterator_dict["high_fidelity_model"]
        lf_model_name = bmfia_iterator_dict["low_fidelity_model"]
        initial_design_dict = bmfia_iterator_dict["initial_design"]

        hf_model = Model.from_config_create_model(hf_model_name, config)
        lf_model = Model.from_config_create_model(lf_model_name, config)

        # ---------- configure external geometry object (returns None if not available) -
        external_geometry = ExternalGeometry.from_config_create_external_geometry(config)

        # ---------- create database object to load coordinates --------------------------
        output_label = config[model_name].get("output_label")
        coord_labels = config[model_name].get("coordinate_labels")
        db = MongoDB.from_config_create_database(config)

        # ---------- calculate the optimal training samples via classmethods ----------
        x_train = cls._calculate_optimal_x_train(initial_design_dict, external_geometry, lf_model)

        # ---------------------- CREATE VISUALIZATION BORG ----------------------------
        # qvis.from_config_create(config)

        return cls(
            result_description,
            global_settings,
            features_config,
            hf_model,
            lf_model,
            output_label,
            coord_labels,
            mf_approx_settings,
            db,
            external_geometry,
            x_train
        )

    @classmethod
    def _calculate_optimal_x_train(cls, initial_design_dict, external_geometry_obj, model):
        """
        Based on the selected design method, determine the optimal set of input points X_train to
        run the HF and the LF model on for the construction of the probabilistic surrogate

        Args:
            initial_design_dict (dict): Dictionary with description of initial design.
            external_geometry_obj (obj): Object with information about an external geometry
            model (obj): A model object on which the calculation is performed (only needed for
                         interfaces here. The model is not evaluated here)

        Returns:
            x_train (np.array): Optimal training input samples

        """
        run_design_method = cls._get_design_method(initial_design_dict)
        x_train = run_design_method(initial_design_dict, external_geometry_obj, model)
        return x_train

    @classmethod
    def _get_design_method(cls, initial_design_dict):
        """
        Get the design method for selecting the HF data from the LF MC data-set

        Args:
            initial_design_dict (dict): Dictionary with description of initial design.

        Returns:
            run_design_method (obj): Design method for selecting the HF training set

        """
        # choose design method
        if initial_design_dict['type'] == 'random':
            run_design_method = cls._random_design
        else:
            raise NotImplementedError

        return run_design_method

    @classmethod
    def _random_design(cls, initial_design_dict, external_geometry_obj, model):
        """
        Calculate the HF training points from large LF-MC data-set based on random selection
        from bins over y_LF.

        Args:
            initial_design_dict (dict): Dictionary with description of initial design.
            external_geometry_obj (obj): Object with external geometry information
            model (obj): A model object on which the calculation is performed (only needed for
                         interfaces here. The model is not evaluated here)

        Returns:
            x_train (np.array): Optimal training input samples

        """
        # Some dummy arguments that are necessary for class initialization but not needed
        dummy_model = model
        dummy_result_description = {}
        dummy_global_settings = {}
        dummy_db = 'dummy_db'

        mc_iterator = MonteCarloIterator(
            dummy_model,
            initial_design_dict['seed'],
            initial_design_dict['num_HF_eval'],
            dummy_result_description,
            dummy_global_settings,
            external_geometry_obj,
            dummy_db,
        )
        mc_iterator.pre_run()
        x_train = mc_iterator.samples
        return x_train

    # ----------- main methods of the object form here ----------------------------------------
    def core_run(self):
        """
        Main or core run of the BMFIA iterator that summarizes the actual evaluation of the HF and
        LF models for these data and the determination of LF informative features.

        Returns:
            Z_train (np.array): Matrix with low-fidelity feature training data
            Y_HF_train (np.array): Matrix with HF training data

        """
        # ----- build model on training points and evaluate it -----------------------
        self.eval_model()

        # ----- Set the feature strategy of the probabilistic mapping (select gammas)
        self._set_feature_strategy()

        return self.Z_train, self.Y_HF_train

    def _evaluate_LF_model_for_X_train(self):
        """
        Evaluate the low-fidelity model for the X_train input data-set

        Returns:
            None

        """
        self.lf_model.update_model_from_sample_batch(self.X_train)
        # reshape the scalar output such that output vector are appended
        self.Y_LF_train = self.lf_model.evaluate()['mean'].reshape(-1, 1)

    def _evaluate_HF_model_for_X_train(self):
        """
        Evaluate the high-fidelity model for the X_train input data-set

        Returns:
            None

        """
        self.hf_model.update_model_from_sample_batch(self.X_train)
        # reshape the scalar output such that output vectors are appended
        self.Y_HF_train = self.hf_model.evaluate()['mean'].reshape(-1, 1)

    def _set_feature_strategy(self):
        """
        Depending on the method specified in the input file, set the strategy that will be used to
        calculate the low-fidelity features :math:`Z_{\\text{LF}}`.

        Returns:
            None

        """
        self.coords_experimental_data = np.tile(
            self.coords_experimental_data, (self.X_train.shape[0], 1)
        )

        if self.settings_probab_mapping['features_config'] == "man_features":
            output_size = int(self.Y_LF_train.shape[0] / self.num_training)
            idx_vec = self.settings_probab_mapping['X_cols']
            gammas_train = np.atleast_2d(self.X_train[:, idx_vec])

            gammas_train_rep = np.matlib.repmat(gammas_train, 1, output_size).reshape(
                -1, gammas_train.shape[1]
            )
            self.gammas_train = np.array(gammas_train_rep)
            self.Z_train = np.hstack(
                [self.Y_LF_train, self.gammas_train, self.coords_experimental_data]
            )
        elif self.settings_probab_mapping['features_config'] == "opt_features":
            if self.settings_probab_mapping['num_features'] < 1:
                raise ValueError()
            self._update_probabilistic_mapping_with_features()
        elif self.settings_probab_mapping['features_config'] == "coord_features":
            self.Z_train = np.hstack([self.Y_LF_train, self.coords_experimental_data])
        elif self.settings_probab_mapping['features_config'] == "no_features":
            self.Z_train = self.Y_LF_train
        else:
            raise IOError("Feature space method specified in input file is unknown!")

    def _update_probabilistic_mapping_with_features(self):
        raise NotImplementedError(
            "Optimal features for inverse problems are not yet implemented! Abort..."
        )



    def eval_model(self):
        """
        Evaluate the LF and HF model to for the training inputs X_train.

        Returns:
            None

        """
        # ---- run LF model on X_train (potentially we need to iterate over this and the previous
        # step to determine optimal X_train; for now just one sequence)
        self._evaluate_LF_model_for_X_train()

        # ---- run HF model on X_train
        self._evaluate_HF_model_for_X_train()

    # ------------------- BELOW JUST PLOTTING AND SAVING RESULTS ------------------
    def post_run(self):
        """
        Saving and plotting of the results.

        Returns:
            None
        """
        if self.result_description['write_results'] is True:
            results = process_ouputs(self.output, self.result_description)
            write_results(
                results,
                self.global_settings["output_dir"],
                self.global_settings["experiment_name"],
            )
