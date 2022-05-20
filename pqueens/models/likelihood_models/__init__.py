"""Likelihood models.

This package contains different likelihood models that can be used
QUEENS, to build probabilistic models. A standard use-case are inverse
problems.
"""

import numpy as np

from pqueens.data_processor import from_config_create_data_processor


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
    experimental_data_base_dir = model_options.get("experimental_csv_data_base_dir")
    experiment_name = global_settings["experiment_name"]
    data_processor_name = model_options.get('data_processor')

    # call method to load experimental data
    try:
        # standard initialization for data_processor
        data_processor = from_config_create_data_processor(config, data_processor_name)
    except ValueError:
        # allow short initialization for data_processor
        # only using the 'file_name_identifier'
        file_name_identifier = model_options.get('experimental_file_name_identifier')
        short_config = {
            "data_processor": {
                "type": "csv",
                "file_name_identifier": file_name_identifier,
                "file_options_dict": {
                    "header_row": 0,
                    "index_column": False,
                    "returned_filter_format": "dict",
                    "filter": {"type": "entire_file"},
                },
            },
        }
        data_processor = from_config_create_data_processor(short_config, "data_processor")

    y_obs_vec, coords_mat, time_vec = _get_experimental_data_and_write_to_db(
        data_processor,
        experimental_data_base_dir,
        experiment_name,
        db,
        coord_labels,
        time_label,
        output_label,
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
    csv_data_reader,
    experimental_data_path,
    experiment_name,
    db,
    coordinate_labels,
    time_label,
    output_label,
):
    """Load all experimental data into QUEENS.

    Args:
        csv_data_reader (obj): CSV - data reader object
        experimental_data_path (str): Path to base directory containing
                                      experimental data
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
    experimental_data_dict = csv_data_reader.get_data_from_file(experimental_data_path)

    # potentially scale experimental data and save the results to the database
    # For now we save all data-points to the experimental data slot `1`. This could be
    # extended in the future if we want to read in several different data sources
    db.save(experimental_data_dict, experiment_name, 'experimental_data', '1')

    # arrange the experimental data coordinates
    experimental_coordinates = (
        np.array([experimental_data_dict[coordinate] for coordinate in coordinate_labels]),
    )[0].T

    # get a unique vector of observation times
    if time_label:
        time_vec = np.sort(list(set(experimental_data_dict[time_label])))
    else:
        time_vec = None

    # get the experimental outputs
    y_obs_vec = np.array(experimental_data_dict[output_label]).reshape(
        -1,
    )

    return y_obs_vec, experimental_coordinates, time_vec
