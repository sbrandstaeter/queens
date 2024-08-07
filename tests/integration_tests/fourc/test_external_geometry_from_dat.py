"""Test external geometry module."""

import numpy as np
import pytest

from queens.external_geometry.fourc_dat_geometry import FourcDatExternalGeometry


def test_external_geometry_from_dat(
    third_party_inputs, expected_node_coordinates, expected_surface_topology
):
    """Test if geometry is read in correctly from dat file."""
    dat_input_template = third_party_inputs / "fourc" / "solid_runtime_hex8.dat"

    # Create pre-processing module form config
    preprocessor_obj = FourcDatExternalGeometry(
        input_template=dat_input_template, list_geometric_sets=["DSURFACE 1"]
    )
    preprocessor_obj.main_run()

    # Check if we got the expected results
    assert preprocessor_obj.surface_topology == expected_surface_topology
    assert preprocessor_obj.node_coordinates["node_mesh"] == expected_node_coordinates["node_mesh"]
    np.testing.assert_allclose(
        preprocessor_obj.node_coordinates["coordinates"],
        expected_node_coordinates["coordinates"],
    )


@pytest.fixture(name="expected_surface_topology")
def fixture_expected_surface_topology():
    """Reference surface topology."""
    expected_topology = [
        {
            "node_mesh": [3, 1, 4, 2],
            "surface_topology": [1, 1, 1, 1],
            "topology_name": "DSURFACE 1",
        }
    ]
    return expected_topology


@pytest.fixture(name="expected_node_coordinates")
def fixture_expected_node_coordinates():
    """Reference node coordinates and nodes."""
    node_coordinates = {
        "node_mesh": [1, 2, 3, 4],
        "coordinates": [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
    }
    return node_coordinates
