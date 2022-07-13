"""Data processor.

Extract data from simulation output.
"""


def from_config_create_data_processor(config, data_processor_name):
    """Create DataProcessor object from problem description.

    Args:
        config (dict): input json file with problem description
        data_processor_name (str): Name of the data processor

    Returns:
        data_processor (obj): data_processor object
    """
    from pqueens.utils.import_utils import get_module_attribute
    from pqueens.utils.valid_options_utils import get_option

    from .data_processor_csv_data import DataProcessorCsv
    from .data_processor_ensight import DataProcessorEnsight
    from .data_processor_ensight_interface import DataProcessorEnsightInterfaceDiscrepancy

    data_processor_dict = {
        'csv': DataProcessorCsv,
        'ensight': DataProcessorEnsight,
        'ensight_interface_discrepancy': DataProcessorEnsightInterfaceDiscrepancy,
    }

    data_processor_options = config.get(data_processor_name)
    if not data_processor_options:
        raise ValueError(
            "The 'data processor' options were not found in the input file! "
            f"You specified the data processor name '{data_processor_name}'. Abort..."
        )

    data_processor_type = data_processor_options.get('type')
    if not data_processor_type:
        raise ValueError(
            "The data_processor section did not specify a valid 'type'! "
            f"Valid options are {data_processor_dict.keys()}. Abort..."
        )

    if data_processor_options.get("external_python_module"):
        module_path = data_processor_options["external_python_module"]
        module_attribute = data_processor_type
        data_processor_class = get_module_attribute(module_path, module_attribute)
    else:
        data_processor_class = get_option(data_processor_dict, data_processor_type)

    data_processor = data_processor_class.from_config_create_data_processor(
        config, data_processor_name
    )
    return data_processor
