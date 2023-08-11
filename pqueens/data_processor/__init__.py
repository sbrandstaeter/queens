"""Data processor.

Extract data from simulation output.
"""

VALID_TYPES = {
    'csv': ['pqueens.data_processor.data_processor_csv', 'DataProcessorCsv'],
    'ensight': ['pqueens.data_processor.data_processor_ensight', 'DataProcessorEnsight'],
    'ensight_interface_discrepancy': [
        'pqueens.data_processor.data_processor_ensight_interface',
        'DataProcessorEnsightInterfaceDiscrepancy',
    ],
    'numpy': ['pqueens.data_processor.data_processor_numpy', 'DataProcessorNumpy'],
}
