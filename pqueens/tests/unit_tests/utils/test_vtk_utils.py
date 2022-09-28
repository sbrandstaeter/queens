"""Test vtk utils."""

import numpy as np
import pytest
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkPoints

from pqueens.utils.path_utils import relative_path_from_pqueens
from pqueens.utils.vtk_utils import (
    add_cell_array_to_vtu_object,
    add_point_array_to_vtu_object,
    export_vtu_file,
    get_node_coordinates_from_vtu_file,
    load_vtu_file,
)


@pytest.fixture(name="template_file_path")
def fixture_template_file_path():
    """Path to vtu example file."""
    return relative_path_from_pqueens("tests/unit_tests/external_files/Rectangle_10x10.vtk")


@pytest.fixture(name="cell_data_array", scope='module', params=[(100), (100, 2), (100, 15)])
def fixture_cell_data_array(request):
    """Numpy array for cell data."""
    return np.ones(request.param)


@pytest.fixture(name="cell_data_vtkarray", scope='module', params=[(100), (100, 2), (100, 15)])
def fixture_cell_data_vtkarray(request):
    """Fixture vtkArray for cell data."""
    array = numpy_to_vtk(np.ones(request.param))
    array.SetName("test_name_different_from_the_other")
    return array


@pytest.fixture(name="point_data_array", scope='module', params=[(121), (121, 2), (121, 15)])
def fixture_point_data_array(request):
    """Numpy array for point data."""
    return np.ones(request.param)


@pytest.fixture(name="point_data_vtkarray", scope='module', params=[(121), (121, 2), (121, 15)])
def fixture_point_data_vtkarray(request):
    """Fixture vtkArray for point data."""
    array = numpy_to_vtk(np.ones(request.param))
    array.SetName("test_name_different_from_the_other")
    return array


def test_load_vtu_file_failure():
    """Test if expection is raised if file not found."""
    file = "none/existing/file"
    with pytest.raises(FileNotFoundError):
        load_vtu_file(file)


def test_load_vtu_file(template_file_path):
    """Test if file is loaded correctly."""
    # Test if the number of nodes matches
    assert load_vtu_file(template_file_path).GetNumberOfPoints() == 121


def test_get_node_coordinates_from_vtu_file_vtk(template_file_path):
    """Check if nodes are read in correctly vtk format."""
    vtk_array = get_node_coordinates_from_vtu_file(template_file_path, as_numpy=False)
    assert isinstance(vtk_array, vtkPoints)
    assert vtk_array.GetNumberOfPoints() == 121


def test_get_node_coordinates_from_vtu_file_numpy(template_file_path):
    """Check if nodes are read in correctly numpy format."""
    vtk_array = get_node_coordinates_from_vtu_file(template_file_path)

    assert isinstance(vtk_array, np.ndarray)
    assert vtk_array.shape == (121, 3)
    np.testing.assert_almost_equal(np.max(vtk_array, axis=0), np.array([1, 1, 0]))
    np.testing.assert_almost_equal(np.min(vtk_array, axis=0), np.array([0, 0, 0]))


def test_add_point_array_to_vtu_object_invalid_type(template_file_path):
    """Check if invalid type raises an error."""
    vtu_object = load_vtu_file(template_file_path)
    with pytest.raises(ValueError, match=r"valid options are"):
        add_point_array_to_vtu_object(vtu_object, "test_array", "Not an array")


def test_add_point_array_to_vtu_object_vtk_failure(template_file_path):
    """Check if wrong dimension of the vtkarray raises an error."""
    vtu_object = load_vtu_file(template_file_path)
    data_array = numpy_to_vtk(np.zeros((10, 1)))
    with pytest.raises(ValueError):
        add_point_array_to_vtu_object(vtu_object, "test_array", data_array)


def test_add_point_array_to_vtu_object_numpy_failure(template_file_path):
    """Check if wrong dimension of the array raises an error."""
    vtu_object = load_vtu_file(template_file_path)
    data_array = np.zeros((10, 1))
    with pytest.raises(ValueError):
        add_point_array_to_vtu_object(vtu_object, "test_array", data_array)


def test_add_point_array_to_vtu_object_numpy(template_file_path, point_data_array):
    """Test if adding a numpy array works."""
    vtu_object = load_vtu_file(template_file_path)
    add_point_array_to_vtu_object(vtu_object, "test_array", point_data_array)
    resulting_array = vtk_to_numpy(vtu_object.GetPointData().GetArray("test_array"))
    np.testing.assert_equal(point_data_array, resulting_array)

    # Check if the name was successfully changed
    not_existing_array = vtu_object.GetPointData().GetArray("test_name_different_from_the_other")
    assert not_existing_array is None


def test_add_point_array_to_vtu_object_vtkarray(template_file_path, point_data_vtkarray):
    """Test if adding a vtkArray works."""
    vtu_object = load_vtu_file(template_file_path)
    add_point_array_to_vtu_object(vtu_object, "test_array", point_data_vtkarray)
    resulting_array = vtu_object.GetPointData().GetArray("test_array")
    assert point_data_vtkarray == resulting_array

    # Check if the name was successfully changed
    not_existing_array = vtu_object.GetPointData().GetArray("test_name_different_from_the_other")
    assert not_existing_array is None


def test_add_cell_array_to_vtu_object_invalid_type(template_file_path):
    """Check if invalid type raises an error."""
    vtu_object = load_vtu_file(template_file_path)
    with pytest.raises(ValueError, match=r"valid options are"):
        add_cell_array_to_vtu_object(vtu_object, "test_array", "Not an array")


def test_add_cell_array_to_vtu_object_vtk_failure(template_file_path):
    """Check if wrong dimension of the vtkarray raises an error."""
    vtu_object = load_vtu_file(template_file_path)
    data_array = numpy_to_vtk(np.zeros((10, 1)))
    with pytest.raises(ValueError):
        add_cell_array_to_vtu_object(vtu_object, "test_array", data_array)


def test_add_cell_array_to_vtu_object_numpy_failure(template_file_path):
    """Check if wrong dimension of the numpy array raises an error."""
    vtu_object = load_vtu_file(template_file_path)
    data_array = np.zeros((10, 1))
    with pytest.raises(ValueError):
        add_cell_array_to_vtu_object(vtu_object, "test_array", data_array)


def test_add_cell_array_to_vtu_object_numpy(template_file_path, cell_data_array):
    """Test if adding a numpy array works."""
    vtu_object = load_vtu_file(template_file_path)
    add_cell_array_to_vtu_object(vtu_object, "test_array", cell_data_array)
    resulting_array = vtk_to_numpy(vtu_object.GetCellData().GetArray("test_array"))
    np.testing.assert_equal(cell_data_array, resulting_array)

    # Check if the name was successfully changed
    not_existing_array = vtu_object.GetCellData().GetArray("test_name_different_from_the_other")
    assert not_existing_array is None


def test_add_cell_array_to_vtu_object_vtkarray(template_file_path, cell_data_vtkarray):
    """Test if adding a vtkArray works."""
    vtu_object = load_vtu_file(template_file_path)
    add_cell_array_to_vtu_object(vtu_object, "test_array", cell_data_vtkarray)
    resulting_array = vtu_object.GetCellData().GetArray("test_array")
    assert cell_data_vtkarray == resulting_array

    # Check if the name was successfully changed
    not_existing_array = vtu_object.GetCellData().GetArray("test_name_different_from_the_other")
    assert not_existing_array is None


def test_export_vtu_file(template_file_path, tmp_path):
    """Test exporting vtu file."""
    vtu_input = load_vtu_file(template_file_path)
    path = tmp_path.joinpath("test.vtu")
    add_cell_array_to_vtu_object(vtu_input, "test", 2 * np.ones((100, 1)))
    export_vtu_file(vtu_input, path)
    vtu_export = load_vtu_file(path)
    cell_data = vtk_to_numpy(vtu_export.GetCellData().GetArray("test"))

    assert path.exists()
    np.testing.assert_almost_equal(cell_data, 2 * np.ones(100))
