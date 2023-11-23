"""Data processor.

Extract data from simulation output.
"""

from queens.data_processor.data_processor_csv import DataProcessorCsv
from queens.data_processor.data_processor_ensight import DataProcessorEnsight
from queens.data_processor.data_processor_ensight_interface import (
    DataProcessorEnsightInterfaceDiscrepancy,
)
from queens.data_processor.data_processor_numpy import DataProcessorNumpy

VALID_TYPES = {
    'csv': DataProcessorCsv,
    'ensight': DataProcessorEnsight,
    'ensight_interface_discrepancy': DataProcessorEnsightInterfaceDiscrepancy,
    'numpy': DataProcessorNumpy,
}
