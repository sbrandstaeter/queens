import abc
import numpy as np
import os
import glob
import pprint
import pandas as pd
from pqueens.models.model import Model
from pqueens.database.mongodb import MongoDB


class LikelihoodModel(Model):
    """
    Base class for likelihood models that unifies interfaces of likelihood models
    used in inverse analysis.

    Attributes:
        forward_model (obj): Forward model on which the likelihood model is based
        coords_mat (np.array): Row-wise coordinates at which the observations were recorded
        y_obs_vec (np.array): Corresponding experimental data vector to coords_mat
        output_label (str): Name of the experimental outputs (column label in csv-file)
        coord_labels (lst): List with coordinate labels for (column labels in csv-file)

    Returns:
        Instance of LikelihoodModel class

    """

    def __init__(
        self,
        model_name,
        model_parameters,
        forward_model,
        coords_mat,
        y_obs_vec,
        output_label,
        coord_labels,
    ):
        super(LikelihoodModel, self).__init__(model_name, model_parameters)
        self.forward_model = forward_model
        self.coords_mat = coords_mat
        self.y_obs_vec = y_obs_vec
        self.output_label = output_label
        self.coord_labels = coord_labels

    @classmethod
    def from_config_create_model(cls, model_name, config):
        """
        Create a likelihood model from the problem description

        Args:
            model_name (str): Name of the model
            config (dict): Dictionary with the problem description

        Returns:
            Instance of likelihood_model class

        """
        # get child likelihood classes
        from .gaussian_static_likelihood import GaussianStaticLikelihood
        from .bayesian_mf_gaussian_static_likelihood import BMFGaussianStaticModel

        model_dict = {
            'gaussian_static': GaussianStaticLikelihood,
            'bmf_gaussian_static': BMFGaussianStaticModel,
        }

        # get options
        model_options = config[model_name]
        model_class = model_dict[model_options["subtype"]]

        forward_model_name = model_options.get("forward_model")
        forward_model = Model.from_config_create_model(forward_model_name, config)

        parameters = model_options["parameters"]
        model_parameters = config[parameters]

        # get further model options
        output_label = model_options.get('output_label')
        coord_labels = model_options.get('coordinate_labels')
        db = MongoDB.from_config_create_database(config)
        global_settings = config.get('global_settings', None)
        experimental_data_path_list = model_options.get("experimental_csv_data_base_dirs")
        experiment_name = global_settings["experiment_name"]

        # call classmethod to load experimental data
        y_obs_vec, coords_mat = cls._get_experimental_data_and_write_to_db(
            experimental_data_path_list, experiment_name, db, coord_labels, output_label
        )

        return model_class.from_config_create_likelihood(
            model_name,
            config,
            model_parameters,
            forward_model,
            coords_mat,
            y_obs_vec,
            output_label,
            coord_labels,
        )

    @classmethod
    def _get_experimental_data_and_write_to_db(
        cls, experimental_data_path_list, experiment_name, db, coordinate_labels, output_label
    ):
        """
        Load all experimental data into QUEENS and conduct some preprocessing and cleaning.

        Args:
            experimental_data_path_list (lst): List containing paths to base directories of
                                               experimental data in csv format
            experiment_name (str): Name of the current experiment in QUEENS
            db (obj): Database object
            coordinate_labels (lst): List of column-wise coordinate labels in csv files
            output_label (str): Label that marks the output quantity in the csv file

        Returns:
            y_obs_vec (np.array): Column-vector of model outputs which correspond row-wise to
                                  observation coordinates
            experimental_coordinates (np.array): Matrix with observation coordinates. One row
                                                 corresponds to one coordinate point.

         """
        if experimental_data_path_list is not None:

            # iteratively load all csv files in specified directory
            files_of_interest_list = []
            all_files_list = []
            for experimental_data_path in experimental_data_path_list:
                prefix_expr = '*.csv'  # only read csv files
                files_of_interest_paths = os.path.join(experimental_data_path, prefix_expr)
                files_of_interest_list.extend(glob.glob(files_of_interest_paths))
                all_files_path = os.path.join(experimental_data_path, '*')
                all_files_list.extend(glob.glob(all_files_path))

            #  check if some files are not csv files and throw a warning
            non_csv_files = [x for x in all_files_list if x not in files_of_interest_list]
            if non_csv_files:
                print('#####################################################################')
                pprint.pprint(
                    f'The following experimental data files could not be read-in as they do '
                    f'not have a .csv file-ending: {non_csv_files}'
                )
                print('#####################################################################')

            # read all experimental data into one numpy array
            # TODO in the future we should use xarrays here
            # TODO filter out / handle corrupted data and NaNs
            data_list = []
            for filename in files_of_interest_list:
                try:
                    new_experimental_data = pd.read_csv(
                        filename, sep=r'[,\s]\s*', header=0, engine='python', index_col=None
                    )
                    data_list.append(new_experimental_data)

                except IOError:
                    raise IOError(
                        'An error occurred while reading in the experimental data '
                        'files. Abort...'
                    )
            experimental_data_dict = pd.concat(data_list, axis=0, ignore_index=True).to_dict('list')

            # potentially scale experimental data and save the results to the database
            # For now we save all data-points to the experimental data slot `1`. This could be
            # extended in the future if we want to read in several different data sources
            db.save(experimental_data_dict, experiment_name, 'experimental_data', '1')

            # arrange the experimental data coordinates
            experimental_coordinates = (
                np.array([experimental_data_dict[coordinate] for coordinate in coordinate_labels]),
            )[0].T
            # get the experimental outputs
            y_obs_vec = np.array(experimental_data_dict[output_label]).squeeze()

            return y_obs_vec, experimental_coordinates

        else:
            raise IOError("You did not specify any experimental data!")

    @abc.abstractmethod
    def evaluate(self):
        """ Evaluate model with current set of variables """
        pass
