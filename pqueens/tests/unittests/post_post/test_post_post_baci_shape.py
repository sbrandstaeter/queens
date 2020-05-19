"""
for distance to surface measurement post_post evaluation

"""

import numpy as np
import pytest

import vtk

import pqueens.post_post.post_post_baci_shape

from pqueens.post_post.post_post_baci_shape import PostPostBACIShape

############## fixtures
@pytest.fixture(scope='module', params=['cut_fsi', '2d_full', '3d_full'])
def all_case_types(request):
    return request.param


@pytest.fixture()
def default_ppbacishapeclass(mocker):
    mocker.patch(
        'pqueens.post_post.post_post_baci_shape.PostPostBACIShape.read_monitorfile',
        return_value='None',
    )
    pp = pqueens.post_post.post_post_baci_shape.PostPostBACIShape(
        'dummy_path', 1e-03, '3d_full', False, True, 'dummy_prefix'
    )
    return pp


@pytest.fixture()
def vtkUnstructuredGridExample2d():

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


def test_init(mocker):

    path_ref_data = 'dummypath'
    time_tol = 1e-03
    case_type = 'dummycase'
    visualization = False
    file_prefix = 'dummyprefix'
    delete_field_data = False

    mp = mocker.patch('pqueens.post_post.post_post.PostPost.__init__')

    mocker.patch(
        'pqueens.post_post.post_post_baci_shape.PostPostBACIShape.read_monitorfile',
        return_value='dummypath',
    )

    my_postpost = PostPostBACIShape(
        path_ref_data, time_tol, case_type, visualization, delete_field_data, file_prefix,
    )

    mp.assert_called_once_with(delete_field_data, file_prefix)

    assert my_postpost.path_ref_data == path_ref_data
    assert my_postpost.time_tol == time_tol
    assert my_postpost.case_type == case_type
    assert my_postpost.visualizationon == visualization
    assert my_postpost.file_prefix == file_prefix


def test_from_config_create_post_post(mocker):
    mp = mocker.patch(
        'pqueens.post_post.post_post_baci_shape.PostPostBACIShape.__init__', return_value=None
    )

    mypostpostoptions = {
        'path_to_ref_data': 'dummypath',
        'time_tol': 1e-03,
        'case_type': 'dummycase',
        'visualization': False,
        'file_prefix': 'dummyprefix',
        'delete_field_data': False,
    }
    myoptions = {'options': mypostpostoptions}

    PostPostBACIShape.from_config_create_post_post(myoptions)
    mp.assert_called_once_with(
        'dummypath', 1e-03, 'dummycase', False, False, 'dummyprefix',
    )


def test_read_post_files(mocker, all_case_types):

    mocker.patch(
        'pqueens.post_post.post_post_baci_shape.PostPostBACIShape.read_monitorfile',
        return_value='None',
    )
    pp = pqueens.post_post.post_post_baci_shape.PostPostBACIShape(
        'dummy_path', 1e-03, all_case_types, False, True, 'dummy_prefix'
    )
    pp.output_dir = 'None'
    mocker.patch('glob.glob', return_value=['None'])
    mocker.patch(
        'pqueens.post_post.post_post_baci_shape.PostPostBACIShape.create_mesh_and_intersect_vtk',
        return_value=[1, 0, 33.98],
    )
    pp.read_post_files()
    assert pp.result == [1, 0, 33.98]

    ##test for ValueErrors

    with pytest.raises(ValueError):
        ppErr = pqueens.post_post.post_post_baci_shape.PostPostBACIShape(
            'dummy_path', 1e-03, 'norealcase', False, True, 'dummy_prefix'
        )
        ppErr.output_dir = 'None'
        ppErr.read_post_files()

    with pytest.raises(ValueError):
        mocker.patch('glob.glob', return_value=['None', 'Neither'])
        pp.read_post_files()


def test_delete_field_data(mocker, default_ppbacishapeclass):

    mp = mocker.patch(
        'pqueens.post_post.post_post_baci_shape.PostPostBACIShape.run_subprocess',
        return_value=[1, 2, 3],
    )
    mocker.patch('os.path.join', return_value='files_search')
    mocker.patch('glob.glob', return_value=['file1', 'file2'])

    default_ppbacishapeclass.delete_field_data()

    mp.assert_any_call('rm file1',)
    mp.assert_any_call('rm file2',)


def test_error_handling(mocker, default_ppbacishapeclass):
    mp = mocker.patch(
        'pqueens.post_post.post_post_baci_shape.PostPostBACIShape.run_subprocess',
        return_value=[1, 2, 3],
    )
    default_ppbacishapeclass.error = True
    default_ppbacishapeclass.output_dir = 'None'
    default_ppbacishapeclass.error_handling()
    mp.assert_called_once_with(
        'cd None&& cd ../.. && mkdir -p postpost_error && cd None&& cd .. && mv *.dat ../postpost_error/',
    )


def test_read_monitorfile(mocker,):

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

    ref_data_compare = [[[0]]]

    mp = mocker.patch('builtins.open', mocker.mock_open(read_data=monitor_string),)

    pp = pqueens.post_post.post_post_baci_shape.PostPostBACIShape(
        'dummy_path', 1e-03, '3d_full', False, True, 'dummy_prefix'
    )

    mp.assert_called_once()

    assert pp.ref_data == [
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
    mp = mocker.patch('builtins.open', mocker.mock_open(read_data=monitor_string),)
    with pytest.raises(ValueError):
        pp.read_monitorfile()


def test_create_mesh_and_intersect_vtk(
    mocker, default_ppbacishapeclass, vtkUnstructuredGridExample2d, vtkUnstructuredGridExample3d
):

    default_ppbacishapeclass.case_type = '2d_full'

    mp = mocker.patch(
        'pqueens.post_post.post_post_baci_shape.PostPostBACIShape.create_UnstructuredGridFromEnsight',
        return_value=vtkUnstructuredGridExample2d,
    )
    default_ppbacishapeclass.ref_data = [
        [1.0, [[[1.0, 1.0, 0], [3.0, 1.0, 0]], [[0.5, 4.0, 0], [0.5, 3.0, 0]],],],
        [2.0, [[[1.0, 1.0, 0], [3.0, 1.0, 0]], [[0.5, 4.0, 0], [0.5, 3.0, 0]],],],
    ]

    assert default_ppbacishapeclass.create_mesh_and_intersect_vtk('dummypath') == [
        -3.0,
        2.0,
        -3.0,
        2.0,
    ]

    default_ppbacishapeclass.case_type = '3d_full'

    mp = mocker.patch(
        'pqueens.post_post.post_post_baci_shape.PostPostBACIShape.create_UnstructuredGridFromEnsight',
        return_value=vtkUnstructuredGridExample3d,
    )

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


def test_stretch_vector(default_ppbacishapeclass):

    assert default_ppbacishapeclass.stretch_vector([1, 2, 3], [2, 4, 6], 2) == [
        [-1, -2, -3],
        [4, 8, 12],
    ]


def test_compute_distance(default_ppbacishapeclass):

    assert default_ppbacishapeclass.compute_distance(
        [[2, 4, 6], [1, 2, 3], [3, 6, 9]], [[0, 0, 0], [0.1, 0.2, 0.3]]
    ) == pytest.approx(np.sqrt(14), abs=10e-12)


def test_create_UnstructuredGridFromEnsight(default_ppbacishapeclass):
    # this method is basically vtk functions only. not really testable unless we mock every line
    pass
