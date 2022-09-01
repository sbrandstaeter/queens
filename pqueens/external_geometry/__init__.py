"""External geometry module.

Read in external geometry to QUEENS.
"""
from pqueens.utils.import_utils import get_module_class

VALID_TYPES = {
    'baci_dat': ["pqueens.external_geometry.baci_dat_geometry", "BaciDatExternalGeometry"]
}


def from_config_create_external_geometry(config, geometry_name):
    """Construct the external_geometry_obj object from the problem description.

    Args:
        config (dict): Dictionary containing the problem description of the QUEENS simulation
        geometry_name (str): Name of the geometry section in input file

    Returns:
        geometry_obj (obj): Instance of the ExternalGeometry class.
    """
    geometry_options = config.get(geometry_name)
    if geometry_options:
        geometry_class = get_module_class(geometry_options, VALID_TYPES)
        geometry_obj = geometry_class.from_config_create_external_geometry(config, geometry_name)
    else:
        geometry_obj = None
    return geometry_obj
