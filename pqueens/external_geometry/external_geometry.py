import abc


class ExternalGeometry(metaclass=abc.ABCMeta):
    """
    Abstract base class to read in external external_geometry_obj formats into QUEENS.
    The class enables, e.g., external_geometry_obj based construction of random fields or post
    processing routines

    Returns:
        geometry_obj (obj): Instance of the ExternalGeometry class.

    """

    def __init__(self):
        pass

    @classmethod
    def from_config_create_external_geometry(cls, config):
        """
        Construct the external_geometry_obj object from the problem description.

        Args:
            config (dict): Dictionary containing the problem description of the QUEENS simulation

        Returns:
            geometry_obj (obj): Instance of the ExternalGeometry class.

        """

        from pqueens.external_geometry.baci_dat_geometry import BaciDatExternalGeometry

        geometry_dict = {
            'baci_dat': BaciDatExternalGeometry,
        }

        geometry = config.get('external_geometry')
        if geometry is not None:
            geometry_version = geometry.get('type')
            geometry_class = geometry_dict[geometry_version]
            # create specific driver
            geometry_obj = geometry_class.from_config_create_external_geometry(config)
        else:
            geometry_obj = None

        return geometry_obj

    def main_run(self):
        """
        Main routine of external_geometry_obj object

        Returns:
            None

        """
        self.organize_sections()
        self.read_external_data()
        self.finish_and_clean()

    @abc.abstractmethod
    def read_external_data(self):
        """
        Method that reads in external files containing a external_geometry_obj definition
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
