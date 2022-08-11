"""External geometry module.

Read in external geometry to QUEENS.
"""
from pqueens.utils.import_utils import get_module_class


def from_config_create_external_geometry(config, geometry_name):
    """Construct the external_geometry_obj object from the problem description.

    Args:
        config (dict): Dictionary containing the problem description of the QUEENS simulation
        geometry_name (str): Name of the geometry section in input file

    Returns:
        geometry_obj (obj): Instance of the ExternalGeometry class.
    """
    valid_types = {
        'baci_dat': [".baci_dat_geometry", "BaciDatExternalGeometry"],
    }

    geometry_options = config.get(geometry_name)
    if geometry_options:
        geometry_type = geometry_options.get("type")
        geometry_class = get_module_class(geometry_options, valid_types, geometry_type)
        geometry_obj = geometry_class.from_config_create_external_geometry(config, geometry_name)
    else:
        geometry_obj = None

    return geometry_obj
