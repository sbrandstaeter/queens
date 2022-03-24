"""External geometry module.

Read in external geometry to QUEENS.
"""


def from_config_create_external_geometry(config):
    """Construct the external_geometry_obj object from the problem description.

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
