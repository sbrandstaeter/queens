"""External geometry module.

Read in external geometry to QUEENS.
"""
from pqueens.utils.import_utils import get_module_attribute
from pqueens.utils.valid_options_utils import get_option


def from_config_create_external_geometry(config, geometry_name):
    """Construct the external_geometry_obj object from the problem description.

    Args:
        config (dict): Dictionary containing the problem description of the QUEENS simulation
        geometry_name (str): Name of the geometry section in input file

    Returns:
        geometry_obj (obj): Instance of the ExternalGeometry class.
    """
    from pqueens.external_geometry.baci_dat_geometry import BaciDatExternalGeometry

    geometry_dict = {
        'baci_dat': BaciDatExternalGeometry,
    }

    geometry_options = config.get(geometry_name)
    if geometry_options:
        if geometry_options.get("external_python_module"):
            module_path = geometry_options["external_python_module"]
            module_attribute = geometry_options.get("type")
            geometry_class = get_module_attribute(module_path, module_attribute)
        elif geometry_options:
            geometry_class = get_option(geometry_dict, geometry_options.get("type"))

        geometry_obj = geometry_class.from_config_create_external_geometry(config, geometry_name)
    else:
        geometry_obj = None

    return geometry_obj
