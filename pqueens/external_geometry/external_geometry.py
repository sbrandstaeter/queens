import abc


class ExternalGeometry(metaclass=abc.ABCMeta):
    """
    Abstract base class to read in external geometry formats into QUEENS.
    The class enables, e.g., geometry based construction of random fields or post processing
    routines

    Returns:
        geometry_obj (obj): Instance of the ExternalGeometry class.

    """

    def __init__(self):
        pass

    @classmethod
    def from_config_create_external_geometry(cls, config):
        """
        Construct the geometry object from the problem description.

        Args:
            config (dict): Dictionary containing the problem description of the QUEENS simulation

        Returns:
            geometry_obj (obj): Instance of the ExternalGeometry class.

        """

        from pqueens.external_geometry.baci_dat_geometry import BaciDatExternalGeometry

        geometry_dict = {
            'baci_dat': BaciDatExternalGeometry,
        }

        geometry_version = config['external_geometry']['type']
        geometry_class = geometry_dict[geometry_version]

        # create specific driver
        geometry_obj = geometry_class.from_config_create_external_geometry(config)

        return geometry_obj

    def main_run(self):
        """
        Main routine of geometry object

        Returns:
            None

        """
        self.organize_sections()
        self.read_external_data()
        self.finish_and_clean()

    @abc.abstractmethod
    def read_external_data(self):
        """
        Method that reads in external files containing a geometry definition
        """
        pass

    @abc.abstractmethod
    def organize_sections(self):
        """
        Organizes (geometric) sections in external file to read in geometric data efficiently
        """
        pass

    @abc.abstractmethod
    def finish_and_clean(self):
        """
        Finishing, postprocessing and cleaning
        """
        pass
