"""Data processor.

Extract data from simulation output.
"""


def from_config_create_data_processor(config, driver_name):
    """Create DataProcessor object from problem description.

    Args:
        config (dict): input json file with problem description
        driver_name (str): Name of driver that is used in this job-submission

    Returns:
        data_processor (obj): data_processor object
    """
    from .data_processor_csv_data import DataProcessorCsv
    from .data_processor_ensight import DataProcessorEnsight
    from .data_processor_ensight_interface import DataProcessorEnsightInterfaceDiscrepancy

    data_processor_dict = {
        'csv': DataProcessorCsv,
        'ensight': DataProcessorEnsight,
        'ensight_interface_discrepancy': DataProcessorEnsightInterfaceDiscrepancy,
    }

    driver_params = config.get(driver_name)
    if not driver_params:
        raise ValueError(
            "No driver parameters found in problem description! "
            f"You specified the driver name '{driver_name}', "
            "which could not be found in the problem description. Abort..."
        )

    try:
        data_processor_options = driver_params["driver_params"].get('data_processor')
    except KeyError:
        data_processor_options = driver_params.get('data_processor')
    if not data_processor_options:
        raise ValueError(
            f"The 'data_processor' options were not found in the driver '{driver_name}'! Abort..."
        )

    data_processor_type = data_processor_options.get('type')
    if not data_processor_type:
        raise ValueError(
            "The data_processor section did not specify a valid 'type'! "
            f"Valid options are {data_processor_dict.keys()}. Abort..."
        )

    data_processor_class = data_processor_dict[data_processor_type]
    data_processor = data_processor_class.from_config_create_data_processor(config, driver_name)
    return data_processor
