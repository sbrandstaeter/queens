"""Data processor.

Extract data from simulation output.
"""
from pqueens.utils.import_utils import get_module_class

VALID_TYPES = {
    'csv': ['pqueens.data_processor.data_processor_csv_data', 'DataProcessorCsv'],
    'ensight': ['pqueens.data_processor.data_processor_ensight', 'DataProcessorEnsight'],
    'ensight_interface_discrepancy': [
        'pqueens.data_processor.data_processor_ensight_interface',
        'DataProcessorEnsightInterfaceDiscrepancy',
    ],
    'numpy': ['pqueens.data_processor.data_processor_numpy', 'DataProcessorNumpy'],
}


def from_config_create_data_processor(config, data_processor_name):
    """Create DataProcessor object from problem description.

    Args:
        config (dict): Input json file with problem description
        data_processor_name (str): Name of the data processor

    Returns:
        data_processor (obj): *data_processor* object
    """
    data_processor_options = config.get(data_processor_name)
    if not data_processor_options:
        raise ValueError(
            "The 'data processor' options were not found in the input file! "
            f"You specified the data processor name '{data_processor_name}'. Abort..."
        )

    data_processor_options = config[data_processor_name]
    data_processor_class = get_module_class(data_processor_options, VALID_TYPES)
    data_processor = data_processor_class.from_config_create_data_processor(
        config, data_processor_name
    )
    return data_processor
