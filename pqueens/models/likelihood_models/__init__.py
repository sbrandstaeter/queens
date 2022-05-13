"""Likelihood models.

This package contains different likelihood models that can be used
QUEENS, to build probabilistic models. A standard use-case are inverse
problems.
"""

import glob
import logging
import os

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)


def from_config_create_model(model_name, config):
    """Create a likelihood model from the problem description.

    Args:
        model_name (str): Name of the model
        config (dict): Dictionary with the problem description

    Returns:
        likelihood_model (obj): Instance of likelihood_model class
    """
    # some other imports
    import pqueens.database.database as DB_module
    from pqueens.models import from_config_create_model

    # get child likelihood classes
    from .bayesian_mf_gaussian_static_likelihood import BMFGaussianStaticModel
    from .gaussian_static_likelihood import GaussianStaticLikelihood

    model_dict = {
        'gaussian_static': GaussianStaticLikelihood,
        'bmf_gaussian_static': BMFGaussianStaticModel,
    }

    # get options
    model_options = config[model_name]
    model_class = model_dict[model_options["subtype"]]

    forward_model_name = model_options.get("forward_model")
    forward_model = from_config_create_model(forward_model_name, config)

    parameters = model_options["parameters"]
    model_parameters = config[parameters]

    # get further model options
    output_label = model_options.get('output_label')
    coord_labels = model_options.get('coordinate_labels')
    time_label = model_options.get('time_label')
    db = DB_module.database
    global_settings = config.get('global_settings', None)
    experimental_data_path_list = model_options.get("experimental_csv_data_base_dirs")
    experiment_name = global_settings["experiment_name"]

    # call classmethod to load experimental data
    y_obs_vec, coords_mat, time_vec = _get_experimental_data_and_write_to_db(
        experimental_data_path_list, experiment_name, db, coord_labels, time_label, output_label
    )

    likelihood_model = model_class.from_config_create_likelihood(
        model_name,
        config,
        model_parameters,
        forward_model,
        coords_mat,
        time_vec,
        y_obs_vec,
        output_label,
        coord_labels,
    )

    return likelihood_model


def _get_experimental_data_and_write_to_db(
    experimental_data_path_list,
    experiment_name,
    db,
    coordinate_labels,
    time_label,
    output_label,
):
    """Load all experimental data into QUEENS.

    Args:
        experimental_data_path_list (lst): List containing paths to base directories of
                                            experimental data in csv format
        experiment_name (str): Name of the current experiment in QUEENS
        db (obj): Database object
        coordinate_labels (lst): List of column-wise coordinate labels in csv files
        time_label (str): Name of the time variable in csv file
        output_label (str): Label that marks the output quantity in the csv file

    Returns:
        y_obs_vec (np.array): Column-vector of model outputs which correspond row-wise to
                                observation coordinates
        experimental_coordinates (np.array): Matrix with observation coordinates. One row
                                                corresponds to one coordinate point.
    """
    try:
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
            _logger.info('----------------------------------------------------------------')
            _logger.info(
                f'The following experimental data files could not be read-in as they do '
                f'not have a .csv file-ending: {non_csv_files}'
            )
            _logger.info('----------------------------------------------------------------')

        # read all experimental data into one numpy array
        data_list = []
        for filename in files_of_interest_list:
            try:
                new_experimental_data = pd.read_csv(
                    filename, sep=r'[,\s]\s*', header=0, engine='python', index_col=None
                )
                data_list.append(new_experimental_data)

            except IOError as my_error:
                raise IOError(
                    'An error has ocurred while reading the experimental data! Abort...'
                ) from my_error
        experimental_data_dict = pd.concat(data_list, axis=0, ignore_index=True).to_dict('list')

        # potentially scale experimental data and save the results to the database
        # For now we save all data-points to the experimental data slot `1`. This could be
        # extended in the future if we want to read in several different data sources
        db.save(experimental_data_dict, experiment_name, 'experimental_data', '1')

        # arrange the experimental data‹ coordinates
        experimental_coordinates = (
            np.array([experimental_data_dict[coordinate] for coordinate in coordinate_labels]),
        )[0].T

        # get a unique vector of observation times
        if time_label:
            time_vec = np.sort(list(set(experimental_data_dict[time_label])))
        else:
            time_vec = None

        # get the experimental outputs
        y_obs_vec = np.array(experimental_data_dict[output_label]).squeeze()

        return y_obs_vec, experimental_coordinates, time_vec

    except IOError as my_error:
        raise IOError("You did not specify any experimental data! Abort...") from my_error
