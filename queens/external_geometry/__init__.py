"""External geometry module.

Read in external geometry to QUEENS.
"""

from queens.external_geometry.baci_dat_geometry import BaciDatExternalGeometry

VALID_TYPES = {
    "baci_dat": BaciDatExternalGeometry,
}
