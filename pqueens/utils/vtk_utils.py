"""Utils to work with vtk."""
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkDoubleArray

from pqueens.utils.path_utils import check_if_path_exists


def load_vtu_file(template_file_path):
    """Load vtu object.

    Args:
        template_file_path (str): Path to file

    Returns:
        vtk object
    """
    check_if_path_exists(template_file_path, "Could not find vtu-file:")
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(template_file_path)
    reader.Update()
    return reader.GetOutput()


def export_vtu_file(vtu_object, template_file_path):
    """Export vtu object.

    Args:
        vtu_object (vtk object): VTU data to be exported
        template_file_path (str): Path where to store this file
    """
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetInputData(vtu_object)
    writer.SetFileName(template_file_path)
    writer.Write()


def get_node_coordinates_from_vtu_file(template_file_path, as_numpy=True):
    """Get nodes from file.

    Args:
        template_file_path (str): Path to file
        as_numpy (bool, optional): Return the array in numpy array or vtkPoints. Defaults to True.

    Returns:
        node_coordinates: in numpy or vtk array object
    """
    vtu_object = load_vtu_file(template_file_path)
    node_coordinates = vtu_object.GetPoints()
    if as_numpy:
        node_coordinates = vtk_to_numpy(node_coordinates.GetData())

    return node_coordinates


def add_cell_array_to_vtu_object(vtu_object, array_name, array_data):
    """Add cell data array to vtu object.

    Args:
        vtu_object (obj): vtk object to which the data should be added
        array_name (str): Array name
        array_data (np.array or vtkArray): data to add to the vtu_object
    """
    vtk_array = array_data
    number_of_points = vtu_object.GetNumberOfCells()
    if isinstance(array_data, np.ndarray):
        vtk_array = numpy_to_vtk(array_data)
        if len(array_data) != number_of_points:
            raise ValueError(
                f"The provided array has dimension {len(array_data)}, but the geometry requires",
                f"{number_of_points} cells.",
            )
    elif isinstance(array_data, vtkDoubleArray):
        if vtk_array.GetNumberOfTuples() != number_of_points:
            raise ValueError(
                f"The provided vtkArray has dimension {array_data.GetNumberOfTuples()}, but the"
                f" geometry requires {number_of_points} cells."
            )
    else:
        raise ValueError(
            f"Type {type(array_data)} not allowed, valid options are np.ndarray or vtkDoubleArray"
        )
    vtk_array.SetName(array_name)
    vtu_object.GetCellData().AddArray(vtk_array)


def add_point_array_to_vtu_object(vtu_object, array_name, array_data):
    """Add point data array to vtu object.

    Args:
        vtu_object (obj): vtk object to which the data should be added
        array_name (str): Array name
        array_data (np.array or vtkArray): data to add to the vtu_object
    """
    vtk_array = array_data
    number_of_points = vtu_object.GetNumberOfPoints()
    if isinstance(array_data, np.ndarray):
        vtk_array = numpy_to_vtk(array_data)
        if len(array_data) != number_of_points:
            raise ValueError(
                f"The provided array has dimension {len(array_data)}, but the geometry requires"
                f"{number_of_points} nodes."
            )
    elif isinstance(array_data, vtkDoubleArray):
        if vtk_array.GetNumberOfTuples() != number_of_points:
            raise ValueError(
                f"The provided vtkArray has dimension {array_data.GetNumberOfTuples()}, but the"
                f"geometry requires {number_of_points} nodes."
            )
    else:
        raise ValueError(
            f"Type {type(array_data)} not allowed, valid options are np.ndarray or vtkDoubleArray"
        )
    vtk_array.SetName(array_name)
    vtu_object.GetPointData().AddArray(vtk_array)
