"""External geometry module.

Read in external geometry to QUEENS.
"""

from queens.external_geometry.fourc_dat_geometry import FourcDatExternalGeometry

VALID_TYPES = {
    "fourc_dat": FourcDatExternalGeometry,
}
