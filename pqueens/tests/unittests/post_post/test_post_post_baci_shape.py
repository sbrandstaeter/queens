"""Tests for distance to surface measurement post_post evaluation."""

import os

import numpy as np
import pytest
import vtk

import pqueens.post_post.post_post_baci_shape
from pqueens.post_post.post_post_baci_shape import PostPostBACIShape


############## fixtures
@pytest.fixture(scope='module', params=['2d', '3d'])
def all_dimensions(request):
    """Parameterized fixture to select problem dimension."""
    return request.param


@pytest.fixture()
def default_ppbacishapeclass(mocker):
    """Default ensight class for upcoming tests."""
    mocker.patch(
        'pqueens.post_post.post_post_baci_shape.PostPostBACIShape.read_monitorfile',
        return_value='None',
    )
    pp = pqueens.post_post.post_post_baci_shape.PostPostBACIShape(
        'dummy_path',
        1e-03,
        False,
        True,
        'dummy_prefix',
        'dummypostfix',
        ['first_disp', 'second_disp'],
        '5d',
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


############## actual tests


@pytest.mark.unit_tests
def test_init(mocker):
    """Test the init method."""
    path_ref_data = 'dummypath'
    time_tol = 1e-03
    visualization = False
    delete_field_data = False
    file_prefix = 'dummyprefix'
    file_postfix = 'dummypostfix'
    displacement_fields = ['first_disp', 'second_disp']
    problem_dimension = '5d'

    mp = mocker.patch('pqueens.post_post.post_post.PostPost.__init__')

    mocker.patch(
        'pqueens.post_post.post_post_baci_shape.PostPostBACIShape.read_monitorfile',
        return_value='dummypath',
    )

    my_postpost = PostPostBACIShape(
        path_ref_data,
        time_tol,
        visualization,
        delete_field_data,
        file_prefix,
        file_postfix,
        displacement_fields,
        problem_dimension,
    )

    mp.assert_called_once_with(delete_field_data, file_prefix)

    assert my_postpost.path_ref_data == path_ref_data
    assert my_postpost.time_tol == time_tol
    assert my_postpost.visualization_bool == visualization
    assert my_postpost.file_prefix == file_prefix
    assert my_postpost.file_postfix == file_postfix
    assert my_postpost.displacement_fields == displacement_fields
    assert my_postpost.problem_dimension == problem_dimension


@pytest.mark.unit_tests
def test_from_config_create_post_post(mocker):
    """Test the config method."""
    mp = mocker.patch(
        'pqueens.post_post.post_post_baci_shape.PostPostBACIShape.__init__', return_value=None
    )

    mypostpostoptions = {
        'path_to_ref_data': 'dummypath',
        'time_tol': 1e-03,
        'visualization': False,
        'file_prefix': 'dummyprefix',
        'delete_field_data': False,
        'file_postfix': 'dummy.case',
        'displacement_fields': ['first_disp', 'second_disp'],
        'problem_dimension': '5d',
    }
    myoptions = {'options': mypostpostoptions}

    PostPostBACIShape.from_config_create_post_post(myoptions)
    mp.assert_called_once_with(
        'dummypath',
        1e-03,
        False,
        False,
        'dummyprefix',
        'dummy.case',
        ['first_disp', 'second_disp'],
        '5d',
    )


@pytest.mark.unit_tests
def test_read_post_files(mocker):
    """Test reading of ensight files."""
    mocker.patch(
        'pqueens.post_post.post_post_baci_shape.PostPostBACIShape.read_monitorfile',
        return_value='None',
    )
    pp = pqueens.post_post.post_post_baci_shape.PostPostBACIShape(
        'dummy_path',
        1e-03,
        False,
        True,
        'dummy_prefix',
        'dummy.case',
        ['first_disp', 'second_disp'],
        '5d',
    )
    pp.output_dir = 'None'
    mocker.patch(
        'pqueens.post_post.post_post_baci_shape.PostPostBACIShape.create_mesh_and_intersect_vtk',
        return_value=[1, 0, 33.98],
    )
    prefix_expr = '*' + pp.file_prefix + '*'
    files_of_interest = os.path.join(pp.output_dir)
    mocker.patch('glob.glob', return_value=['any'])
    pp.read_post_files(files_of_interest)
    assert pp.result == [1, 0, 33.98]

    ##test for ValueErrors
    with pytest.raises(ValueError):
        ppErr = pqueens.post_post.post_post_baci_shape.PostPostBACIShape(
            'dummy_path',
            1e-03,
            False,
            True,
            'dummy_prefix',
            'dummy.case',
            ['first_disp', 'second_disp'],
            '5d',
        )
        ppErr.output_dir = 'None'
        mocker.patch('glob.glob', return_value=[])
        ppErr.read_post_files(files_of_interest)

    with pytest.raises(ValueError):
        mocker.patch('glob.glob', return_value=['any', 'other'])
        pp.read_post_files(files_of_interest)


@pytest.mark.unit_tests
def test_read_monitorfile(
    mocker,
):
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

    pp = pqueens.post_post.post_post_baci_shape.PostPostBACIShape(
        'dummy_path',
        1e-03,
        False,
        True,
        'dummy_prefix',
        'dummypostfix',
        ['first_disp', 'second_disp'],
        '5d',
    )

    mp.assert_called_once()

    assert pp.experimental_ref_data_lst == [
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
        pp.read_monitorfile()


@pytest.mark.unit_tests
def test_create_mesh_and_intersect_vtk(
    mocker, default_ppbacishapeclass, vtkUnstructuredGridExample2d, vtkUnstructuredGridExample3d
):
    """Test for vtk mesh intersection."""
    default_ppbacishapeclass.problem_dimension = '2d'

    # pylint: disable=line-too-long error
    mp = mocker.patch(
        'pqueens.post_post.post_post_baci_shape.PostPostBACIShape.create_UnstructuredGridFromEnsight',
        return_value=vtkUnstructuredGridExample2d,
    )
    # pylint: enable=line-too-long error
    default_ppbacishapeclass.ref_data = [
        [
            1.0,
            [
                [[1.0, 1.0, 0], [3.0, 1.0, 0]],
                [[0.5, 4.0, 0], [0.5, 3.0, 0]],
            ],
        ],
        [
            2.0,
            [
                [[1.0, 1.0, 0], [3.0, 1.0, 0]],
                [[0.5, 4.0, 0], [0.5, 3.0, 0]],
            ],
        ],
    ]

    assert default_ppbacishapeclass.create_mesh_and_intersect_vtk('dummypath') == [
        -3.0,
        2.0,
        -3.0,
        2.0,
    ]

    default_ppbacishapeclass.problem_dimension = '3d'

    # pylint: disable=line-too-long error
    mp = mocker.patch(
        'pqueens.post_post.post_post_baci_shape.PostPostBACIShape.create_UnstructuredGridFromEnsight',
        return_value=vtkUnstructuredGridExample3d,
    )
    # pylint: enable=line-too-long error

    default_ppbacishapeclass.ref_data = [
        [
            1.0,
            [
                [[-1, 2.5, 1], [0, 1.5, 1]],
                [[0, 0.5, 1], [1, 0.5, 1]],
                [[0.5, 1.25, 0], [0.5, 1.25, 1]],
            ],
        ],
    ]

    assert default_ppbacishapeclass.create_mesh_and_intersect_vtk('dummypath') == pytest.approx(
        [1.767766952966, 0.0, -0.499999999999], abs=10e-10
    )


@pytest.mark.unit_tests
def test_stretch_vector(default_ppbacishapeclass):
    """Test for stretch vector helpre method."""
    assert default_ppbacishapeclass.stretch_vector([1, 2, 3], [2, 4, 6], 2) == [
        [-1, -2, -3],
        [4, 8, 12],
    ]


@pytest.mark.unit_tests
def test_compute_distance(default_ppbacishapeclass):
    """Test for distance computation."""
    assert default_ppbacishapeclass.compute_distance(
        [[2, 4, 6], [1, 2, 3], [3, 6, 9]], [[0, 0, 0], [0.1, 0.2, 0.3]]
    ) == pytest.approx(np.sqrt(14), abs=10e-12)
