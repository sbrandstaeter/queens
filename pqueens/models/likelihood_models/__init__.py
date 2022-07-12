"""Likelihood models.

This package contains different likelihood models that can be used
QUEENS, to build probabilistic models. A standard use-case are inverse
problems.
"""

from pathlib import Path

from pqueens.utils.get_experimental_data import get_experimental_data, write_experimental_data_to_db
from pqueens.utils.import_utils import get_module_attribute


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
    from .bayesian_mf_gaussian_likelihood import BMFGaussianModel
    from .gaussian_likelihood import GaussianLikelihood

    model_dict = {
        'gaussian': GaussianLikelihood,
        'bmf_gaussian': BMFGaussianModel,
    }

    # get options
    model_options = config[model_name]

    if model_options["subtype"] in model_dict.keys():
        model_class = model_dict[model_options["subtype"]]
    elif model_options.get("path_external_python_module"):
        module_path = model_options["path_external_python_module"]
        module_attribute = model_options["subtype"]
        model_class = get_module_attribute(module_path, module_attribute)
    else:
        raise ModuleNotFoundError(
            f"The module '{model_options['subtype']}' could not be found!\n"
            f"Valid internal modules are: {model_dict.keys()}.\n"
            "If you provided an external Python module, make sure you specified \n"
            "the correct class name under the 'type' keyword!"
        )

    forward_model_name = model_options.get("forward_model")
    forward_model = from_config_create_model(forward_model_name, config)

    # get further model options
    output_label = model_options.get('output_label')
    coord_labels = model_options.get('coordinate_labels')
    time_label = model_options.get('time_label')
    db = DB_module.database
    global_settings = config.get('global_settings', None)
    experiment_name = global_settings["experiment_name"]
    data_processor_name = model_options.get('data_processor')
    file_name = model_options.get('experimental_file_name_identifier')
    base_dir = model_options.get('experimental_csv_data_base_dir')

    y_obs_vec, coords_mat, time_vec, experimental_data_dict = get_experimental_data(
        config=config,
        data_processor_name=data_processor_name,
        base_dir=base_dir,
        file_name=file_name,
        coordinate_labels=coord_labels,
        time_label=time_label,
        output_label=output_label,
    )
    write_experimental_data_to_db(experimental_data_dict, experiment_name, db)

    likelihood_model = model_class.from_config_create_likelihood(
        model_name,
        config,
        forward_model,
        coords_mat,
        time_vec,
        y_obs_vec,
        output_label,
        coord_labels,
    )

    return likelihood_model
