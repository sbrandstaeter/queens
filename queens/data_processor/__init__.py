"""Data processor.

Extract data from simulation output.
"""

VALID_TYPES = {
    'csv': ['queens.data_processor.data_processor_csv', 'DataProcessorCsv'],
    'ensight': ['queens.data_processor.data_processor_ensight', 'DataProcessorEnsight'],
    'ensight_interface_discrepancy': [
        'queens.data_processor.data_processor_ensight_interface',
        'DataProcessorEnsightInterfaceDiscrepancy',
    ],
    'numpy': ['queens.data_processor.data_processor_numpy', 'DataProcessorNumpy'],
}
