"""TODO_doc."""

import abc
import logging

_logger = logging.getLogger(__name__)


class ExternalGeometry(metaclass=abc.ABCMeta):
    """Abstract base class to read in *external_geometry_obj* formats.

    The class enables, e.g., *external_geometry_obj* based construction
    of random fields or post processing routines.

    Returns:
        geometry_obj (obj): Instance of the ExternalGeometry class
    """

    def __init__(self):
        """TODO_doc."""

    def main_run(self):
        """Main routine of *external_geometry_obj* object."""
        _logger.info('Start reading external geometry from file...')
        self.organize_sections()
        self.read_external_data()
        self.finish_and_clean()
        _logger.info('Finished reading external geometry from file!')

    @abc.abstractmethod
    def read_external_data(self):
        """Method that reads in external files.

        Method that reads in external files containing an
        *external_geometry_obj* definition.
        """

    @abc.abstractmethod
    def organize_sections(self):
        """Organizes (geometric) sections.

        Organizes (geometric) sections in external file to read in
        geometric data efficiently.
        """

    @abc.abstractmethod
    def finish_and_clean(self):
        """Finishing, postprocessing and cleaning."""
