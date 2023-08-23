"""Test external geometry module."""


import numpy as np
import pytest

from pqueens.external_geometry.baci_dat_geometry import BaciDatExternalGeometry


def test_external_geometry_from_dat(
    third_party_inputs, expected_node_coordinates, expected_surface_topology
):
    """Test if geometry is read in correctly from dat file."""
    dat_input_template = (
        third_party_inputs / "baci_input_files" / "meshtying3D_patch_lin_duallagr_new_struct.dat"
    )

    # Create pre-processing module form config
    preprocessor_obj = BaciDatExternalGeometry(
        input_template=dat_input_template, list_geometric_sets=["DSURFACE 1"]
    )
    preprocessor_obj.main_run()

    # Check if we got the expected results
    assert preprocessor_obj.surface_topology == expected_surface_topology
    assert preprocessor_obj.node_coordinates['node_mesh'] == expected_node_coordinates['node_mesh']
    np.testing.assert_allclose(
        preprocessor_obj.node_coordinates['coordinates'],
        expected_node_coordinates['coordinates'],
        rtol=1.0e-3,
    )


@pytest.fixture(name="expected_surface_topology")
def expected_surface_topology_fixture():
    """Reference surface topology."""
    expected_topology = [
        {
            'node_mesh': [
                145,
                148,
                149,
                152,
                162,
                164,
                170,
                172,
                177,
                180,
                186,
                190,
                193,
                196,
                202,
                206,
            ],
            'surface_topology': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            'topology_name': 'DSURFACE 1',
        }
    ]
    return expected_topology


@pytest.fixture(name="expected_node_coordinates")
def expected_node_coordinates_fixture():
    """Reference node coordinates and nodes."""
    node_coordinates = {
        'node_mesh': [
            145,
            148,
            149,
            152,
            162,
            164,
            170,
            172,
            177,
            180,
            186,
            190,
            193,
            196,
            202,
            206,
        ],
        'coordinates': [
            [-2.5, -2.5, 6.0],
            [-2.5, -0.8333333333333335, 6.0],
            [-0.8333333333333333, -2.5, 6.0],
            [-0.8333333333333333, -0.8333333333333338, 6.0],
            [-2.5, 0.8333333333333333, 6.0],
            [-0.8333333333333335, 0.8333333333333335, 6.0],
            [-2.5, 2.5, 6.0],
            [-0.8333333333333335, 2.5, 6.0],
            [0.8333333333333335, -2.5, 6.0],
            [0.8333333333333335, -0.8333333333333333, 6.0],
            [0.8333333333333334, 0.8333333333333331, 6.0],
            [0.8333333333333333, 2.5, 6.0],
            [2.5, -2.5, 6.0],
            [2.5, -0.8333333333333333, 6.0],
            [2.5, 0.8333333333333335, 6.0],
            [2.5, 2.5, 6.0],
        ],
    }
    return node_coordinates
