"""Tests for distance to surface measurement post_post evaluation."""

import os
from re import I

import numpy as np
import pytest
import vtk

import pqueens.post_post.post_post_ensight_interface
from pqueens.post_post.post_post_ensight_interface import PostPostEnsightInterfaceDiscrepancy


############## fixtures
@pytest.fixture(scope='module', params=['2d', '3d'])
def all_dimensions(request):
    """Parameterized fixture to select problem dimension."""
    return request.param


@pytest.fixture()
def default_post_post(mocker):
    """Default ensight class for upcoming tests."""
    post_post_file_name_prefix = ('dummy_prefix*dummypostfix',)
    file_options_dict = {}
    experimental_ref_data = 'dummy'
    displacement_fields = ['first_disp', 'second_disp']
    time_tol = 1e-03
    visualization_bool = False
    file_to_be_deleted_regex_lst = []
    driver_name = 'driver'
    problem_dim = '5d'

    # pylint: disable=line-too-long error
    mocker.patch(
        'pqueens.post_post.post_post_ensight_interface.PostPostEnsightInterfaceDiscrepancy.read_monitorfile',
        return_value='None',
    )
    # pylint: enable=line-too-long error
    pp = pqueens.post_post.post_post_ensight_interface.PostPostEnsightInterfaceDiscrepancy(
        post_post_file_name_prefix,
        file_options_dict,
        file_to_be_deleted_regex_lst,
        driver_name,
        time_tol,
        visualization_bool,
        displacement_fields,
        problem_dim,
        experimental_ref_data,
    )
    return pp


@pytest.fixture()
def vtkUnstructuredGridExample2d():
    """Exemplary vtk grid."""
    node_coords = [
        [-2, 0, 0],
        [0, 0, 0],
        [2, 0, 0],
        [-2, 2, 0],
        [-1, 2, 0],
        [1, 2, 0],
        [2, 2, 0],
    ]

    grid = vtk.vtkUnstructuredGrid()

    vtkpoints = vtk.vtkPoints()

    for i in range(0, len(node_coords)):
        vtkpoints.InsertPoint(i, node_coords[i])

    grid.InsertNextCell(vtk.VTK_QUAD, 4, [0, 1, 4, 3])
    grid.InsertNextCell(vtk.VTK_QUAD, 4, [1, 2, 6, 5])
    grid.InsertNextCell(vtk.VTK_TRIANGLE, 3, [1, 5, 4])

    grid.SetPoints(vtkpoints)

    return grid


@pytest.fixture()
def vtkUnstructuredGridExample3d():
    """Exemplary unstructured 3D vtk grid."""
    node_coords = [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0.5, 1.5, 0],
        [0, 0, 2],
        [1, 0, 2],
        [1, 1, 2],
        [0, 1, 2],
        [0.5, 1.5, 2],
        [0.5, 1.5, -1],
    ]

    grid = vtk.vtkUnstructuredGrid()

    vtkpoints = vtk.vtkPoints()

    for i in range(0, len(node_coords)):
        vtkpoints.InsertPoint(i, node_coords[i])

    grid.InsertNextCell(vtk.VTK_HEXAHEDRON, 8, [0, 1, 2, 3, 5, 6, 7, 8])
    grid.InsertNextCell(vtk.VTK_WEDGE, 6, [3, 2, 4, 8, 7, 9])
    grid.InsertNextCell(vtk.VTK_TETRA, 4, [2, 3, 4, 10])

    grid.SetPoints(vtkpoints)

    return grid


# --------------- actual tests -------------------------


@pytest.mark.unit_tests
def test_init(mocker):
    """Test the init method."""
    post_post_file_name_prefix = ('dummy_prefix*dummypostfix',)
    file_options_dict = {}
    experimental_ref_data = 'dummy'
    displacement_fields = ['first_disp', 'second_disp']
    time_tol = 1e-03
    visualization_bool = False
    files_to_be_deleted_regex_lst = []
    driver_name = 'driver'
    problem_dim = '5d'

    my_postpost = PostPostEnsightInterfaceDiscrepancy(
        post_post_file_name_prefix,
        file_options_dict,
        files_to_be_deleted_regex_lst,
        driver_name,
        time_tol,
        visualization_bool,
        displacement_fields,
        problem_dim,
        experimental_ref_data,
    )

    assert my_postpost.time_tol == time_tol
    assert my_postpost.visualization_bool is visualization_bool
    assert my_postpost.displacement_fields == displacement_fields
    assert my_postpost.problem_dimension == problem_dim
    assert my_postpost.experimental_ref_data_lst == experimental_ref_data

    assert my_postpost.files_to_be_deleted_regex_lst == files_to_be_deleted_regex_lst
    assert my_postpost.file_options_dict == file_options_dict
    assert my_postpost.driver_name == driver_name
    assert my_postpost.post_post_file_name_prefix == post_post_file_name_prefix
    assert my_postpost.post_post_file_path is None
    np.testing.assert_array_equal(my_postpost.post_post_data, np.empty(shape=0))
    assert my_postpost.raw_file_data is None


@pytest.mark.unit_tests
def test_from_config_create_post_post(mocker):
    """Test the config method."""
    experimental_ref_data = np.array([[1, 2], [3, 4]])
    # pylint: disable=line-too-long error
    mp = mocker.patch(
        'pqueens.post_post.post_post_ensight_interface.PostPostEnsightInterfaceDiscrepancy.__init__',
        return_value=None,
    )

    mp2 = mocker.patch(
        'pqueens.post_post.post_post_ensight_interface.PostPostEnsightInterfaceDiscrepancy.read_monitorfile',
        return_value=experimental_ref_data,
    )
    # pylint: enable=line-too-long error
    driver_name = 'driver'
    post_post_file_name_prefix = 'dummyprefix*dummy.case'
    file_options_dict = {
        'time_tol': 1e-03,
        'visualization': False,
        'displacement_fields': ['first_disp', 'second_disp'],
        'delete_field_data': False,
        'problem_dimension': '5d',
        'path_to_ref_data': 'some_path',
    }
    files_to_be_deleted_regex_lst = []
    config = {}

    PostPostEnsightInterfaceDiscrepancy.from_config_create_post_post(
        driver_name,
        post_post_file_name_prefix,
        file_options_dict,
        files_to_be_deleted_regex_lst,
        config,
    )
    mp.assert_called_once_with(
        post_post_file_name_prefix,
        file_options_dict,
        files_to_be_deleted_regex_lst,
        driver_name,
        1e-3,
        False,
        ['first_disp', 'second_disp'],
        '5d',
        experimental_ref_data,
    )


@pytest.mark.unit_tests
def test_read_monitorfile(mocker):
    """Test reading of monitor file."""
    # monitor_string will be used to mock the content of a monitor file that is linked at
    # path_to_ref_data whereas the indentation is compulsory
    monitor_string = """#somecomment
steps 2 npoints 4
2 0 1
2 0 2
2 1 2
3 0 1 2
#comments here and in following lines
#lines above: #number of dimensions for point pairs #ID of coordinate directions
# following lines in scheme seperated by arbitrary number of spaces
# (first time point) x1 y1 x1' y1' x2 y2 x2' y2' x3 y3 x3' y3' x4 y4 x4' y4' x5 y5 x5' y5'
# (x y) is a location of the interface (x' y') is a point that is associated with
# the direction in which the distance to the interface is measured
# the vectors (x y)->(x' y') should point towards the interface
4.0e+00 1.0 1.0 1.0 1.0  2.0 2.0 2.0 2.0  3.0 3.0 3.0 3.0  1.0 1.0 1.0 1.0 1.0 1.0
8.0e+00 5.0 5.0 5.0 5.0  6.0 6.0 6.0 6.0  7.0 7.0 7.0 7.0  5.0 5.0 5.0 5.0 5.0 5.0"""

    mp = mocker.patch('builtins.open', mocker.mock_open(read_data=monitor_string))
    data = PostPostEnsightInterfaceDiscrepancy.read_monitorfile('dummy_path')
    mp.assert_called_once()

    assert data == [
        [
            4.0,
            [
                [[1.0, 1.0, 0], [1.0, 1.0, 0]],
                [[2.0, 0, 2.0], [2.0, 0, 2.0]],
                [[0, 3.0, 3.0], [0, 3.0, 3.0]],
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            ],
        ],
        [
            8.0,
            [
                [[5.0, 5.0, 0], [5.0, 5.0, 0]],
                [[6.0, 0, 6.0], [6.0, 0, 6.0]],
                [[0, 7.0, 7.0], [0, 7.0, 7.0]],
                [[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]],
            ],
        ],
    ]

    monitor_string = '''something wrong'''
    mocker.patch('builtins.open', mocker.mock_open(read_data=monitor_string))
    with pytest.raises(ValueError):
        PostPostEnsightInterfaceDiscrepancy.read_monitorfile('some_path')


@pytest.mark.unit_tests
def test_stretch_vector(default_post_post):
    """Test for stretch vector helpre method."""
    assert default_post_post._stretch_vector([1, 2, 3], [2, 4, 6], 2) == [
        [-1, -2, -3],
        [4, 8, 12],
    ]


@pytest.mark.unit_tests
def test_compute_distance(default_post_post):
    """Test for distance computation."""
    assert default_post_post._compute_distance(
        [[2, 4, 6], [1, 2, 3], [3, 6, 9]], [[0, 0, 0], [0.1, 0.2, 0.3]]
    ) == pytest.approx(np.sqrt(14), abs=10e-12)
