"""Data processor module for vtk ensight boundary data."""

import re
from pathlib import Path

import numpy as np
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy

from .data_processor import DataProcessor


class DataProcessorEnsightInterfaceDiscrepancy(DataProcessor):
    """Discrepancy measure for boundaries and shapes.

    data_processor class uses full ensight result in vtk to measure distance of
    surface from simulations to experiment.

    Attributes:
        time_tol (float): time tolerance for given reference timepoints
        visualization_bool (bool): boolean for vtk visualization control
        displacement_fields (str): String with exact field names for displacement to apply
        problem_dimension (string): string to determine problems spatial dimension
        experimental_ref_data_lst (list): Experimental reference data to which the
                                          discrepancy measure is computed.
    """

    def __init__(
        self,
        file_name_identifier,
        file_options_dict,
        files_to_be_deleted_regex_lst,
        data_processor_name,
        time_tol,
        visualization_bool,
        displacement_fields,
        problem_dimension,
        experimental_reference_data_lst,
    ):
        """Initialize data_processor_ensight_interface class.

        Args:
            file_name_identifier (str): Identifier of file name.
                                             The file prefix can contain regex expression
                                             and subdirectories.
            file_options_dict (dict): Dictionary with read-in options for
                                      the file
            files_to_be_deleted_regex_lst (lst): List with paths to files that should be deleted.
                                                 The paths can contain regex expressions.
            data_processor_name (str): Name of the data processor.
            time_tol (float): time tolerance for given reference timepoints
            visualization_bool (bool): boolean for vtk visualization control
            displacement_fields (str): String with exact field names for displacement to apply
            problem_dimension (string): string to determine problems spatial dimension
            experimental_reference_data_lst (list): Experimental reference data to which the
                                              discrepancy measure is computed.
        """
        super().__init__(
            file_name_identifier,
            file_options_dict,
            files_to_be_deleted_regex_lst,
            data_processor_name,
        )
        self.time_tol = time_tol
        self.visualization_bool = visualization_bool
        self.displacement_fields = displacement_fields
        self.problem_dimension = problem_dimension
        self.experimental_ref_data_lst = experimental_reference_data_lst

    @classmethod
    def from_config_create_data_processor(cls, config, data_processor_name):
        """Create the class from the problem description.

        Args:
            config (dict): Dictionary with problem description.
            data_processor_name (str): Name of the data processor
        """
        (
            file_name_identifier,
            file_options_dict,
            files_to_be_deleted_regex_lst,
        ) = super().from_config_set_base_attributes(config, data_processor_name)

        path_ref_data_str = file_options_dict.get('path_to_ref_data')
        if not path_ref_data_str:
            raise ValueError(
                "You must provide the option 'path_to_ref_data' within the 'file_options_dict' "
                f"in '{data_processor_name}'. Abort ..."
            )
        path_ref_data = Path(path_ref_data_str)
        experimental_reference_data = cls.read_monitorfile(path_ref_data)

        time_tol = file_options_dict.get('time_tol')
        if not time_tol:
            raise ValueError(
                "You must provide the option 'time_tol' within the 'file_options_dict' "
                f"in '{data_processor_name}'. Abort ..."
            )

        visualization_bool = file_options_dict.get('visualization', False)
        if not isinstance(visualization_bool, bool):
            raise TypeError(
                "The option 'visualization_bool' must be of type 'bool' "
                f"but you provided type {type(visualization_bool)}. Abort..."
            )

        displacement_fields = file_options_dict.get('displacement_fields', ['displacement'])
        if not isinstance(displacement_fields, list):
            raise TypeError(
                "The option 'displacement_fields' must be of type 'list' "
                f"but you provided type {type(displacement_fields)}. Abort..."
            )

        problem_dimension = file_options_dict.get('problem_dimension', '2d')
        if not isinstance(problem_dimension, str):
            raise TypeError(
                "The option 'problem_dimension' must be of type 'str' "
                f"but you provided type {type(problem_dimension)}. Abort..."
            )

        return cls(
            file_name_identifier,
            file_options_dict,
            files_to_be_deleted_regex_lst,
            data_processor_name,
            time_tol,
            visualization_bool,
            displacement_fields,
            problem_dimension,
            experimental_reference_data,
        )

    @staticmethod
    def read_monitorfile(path_to_experimental_reference_data):
        """Read Monitor file.

        The Monitor File contains measurements from the
        experiments.

        Args:
            path_to_experimental_reference_data (path obj):
                path to experimental reference data

        Returns:
            monfile_data (list): data from monitor file in numbers
        """
        with open(path_to_experimental_reference_data) as my_file:
            lines = my_file.readlines()
            i = 0
            npoints = 0
            steps = 0
            # lines specifying number of spatial dimensions and dimension ids
            npoint_lines = []
            # measurements for all points in different time steps
            steps_lines = []
            # sort lines into npoint_lines and steps_lines
            for line in lines:
                if line.startswith('#'):
                    continue
                line = line.strip()
                if line.startswith('steps'):
                    firstline = line
                    steps = re.findall('^(?:steps )(.+)(?= npoints)', firstline, re.M)
                    steps = int(steps[0])
                    npoints = re.findall('^(?:steps )(?:.+)?(?: npoints )(.+)', firstline, re.M)
                    npoints = int(npoints[0])
                    continue
                if i < npoints:
                    npoint_lines.append(line.split())
                    i += 1
                    continue
                if i - npoints - 1 < steps:
                    steps_lines.append(line.split())

            if npoints == 0 or steps == 0:
                raise ValueError(
                    'read_monitorfile did not find useful content. Monitor format is probably wrong'
                )

        # read numeric content from file data
        npoint_lines = [[int(ii) for ii in i] for i in npoint_lines]
        steps_lines = [[float(ii) for ii in i] for i in steps_lines]

        # prefill monfile_data of adequate size with zeros
        # monfile_data has dimensions
        # [number of timesteps][2][number of points][2][3dim]
        # it contains pairs of points on the interface and in the domain (for distance
        # in prescribed direction) measured in experiment
        monfile_data = []
        for i in steps_lines:
            monfile_data.append(
                [[0.0e0], [[[0, 0, 0] for j in range(0, 2)] for k in range(0, npoints)]]
            )

        # for all npoint_lines read according data from steps_lines to monfile_data
        # loop over time steps
        for i in range(len(steps_lines)):
            k = 1
            # save time value for time step
            monfile_data[i][0] = steps_lines[i][0]
            # loop over points
            for ii in range(len(npoint_lines)):
                for x in range(0, 2):
                    for iii in range(0, npoint_lines[ii][0]):
                        monfile_data[i][1][ii][x][npoint_lines[ii][iii + 1]] = steps_lines[i][k]
                        k += 1
        return monfile_data

    def _get_raw_data_from_file(self):
        """Read-in EnSight file using vtkGenericEnSightReader."""
        self.raw_file_data = vtk.vtkGenericEnSightReader()
        self.raw_file_data.SetCaseFileName(self.file_path)
        self.raw_file_data.ReadAllVariablesOn()
        self.raw_file_data.Update()

    def _filter_and_manipulate_raw_data(self):
        """Get deformed boundary from vtk.

        Create vtk representation of deformed external_geometry_obj and
        evaluate surface distance measurement for every given timestep from the
        experiment.

        Returns:
            residual (list): full residual from this data_processor class
        """
        residual_distance_lst = []
        points = vtk.vtkPoints()
        vertices = vtk.vtkCellArray()

        for current_step_experimental_data in self.experimental_ref_data_lst:

            grid = self.create_UnstructuredGridFromEnsight_per_time_step(
                current_step_experimental_data[0]
            )
            geo = vtk.vtkGeometryFilter()
            geo.SetInputData(grid)
            geo.Update()
            geometry_output = geo.GetOutput()
            outline_out, outline_data = self._get_dim_dependent_vtk_output(geometry_output)

            for measured_point_pair in current_step_experimental_data[1]:
                point_vector = self._stretch_vector(
                    measured_point_pair[0], measured_point_pair[1], 10
                )
                intersection_points = self._get_intersection_points(
                    outline_data, outline_out, point_vector
                )
                distance = self._compute_distance(intersection_points, measured_point_pair)
                residual_distance_lst.append(distance)

                self._visualize_intermediate_discrepancy_measure(
                    points, vertices, intersection_points, point_vector
                )

            self._visualize_final_discrepancy_measure(outline_out, points, vertices)

        self.processed_data = residual_distance_lst

    def _get_intersection_points(self, outline_data, outline_out, point_vector):
        """Get intersection points."""
        counter = 0
        intersection_points_lst = []

        while counter < len(outline_data):
            numpoints = outline_data.item(counter)
            locpointids = outline_data[counter + 1 : counter + 1 + numpoints]

            locations = []
            for idx in np.nditer(locpointids):
                x = [0, 0, 0]
                outline_out.GetPoint(idx, x)
                locations.append(x)

            local_points = vtk.vtkPoints()
            for location in locations:
                local_points.InsertNextPoint(location)

            local_element = self._get_local_element(locations)
            local_element.Initialize(len(locations), local_points)

            intersection_point = [0, 0, 0]
            pcoords = [0, 0, 0]

            # key line of the algorithm: line intersection
            intersectionfound = local_element.IntersectWithLine(
                point_vector[0],
                point_vector[1],
                1e-12,
                vtk.reference(0),
                intersection_point,
                pcoords,
                vtk.reference(0),
            )

            if intersectionfound:
                intersection_points_lst.append(intersection_point)

            counter += numpoints
            counter += 1

        return intersection_points_lst

    def _get_local_element(self, locations):
        """Get the local element based on input dimension."""
        if len(locations) == 2:
            local_element = vtk.vtkLine()
        elif len(locations) == 3:
            local_element = vtk.vtkTriangle()
        elif len(locations) == 4:
            local_element = vtk.vtkQuad()
        else:
            raise ValueError('Unknown local_element type for structure surface discretization.')
        return local_element

    def _get_dim_dependent_vtk_output(self, geoout):
        """Return the vtk output dependent of problem dimension."""
        if self.problem_dimension == '2d':
            outline = vtk.vtkFeatureEdges()
            outline.SetInputData(geoout)
            outline.Update()
            outlineout = outline.GetOutput()

            outlines = outlineout.GetLines()
            outline_data_vtk = outlines.GetData()

        elif self.problem_dimension == '3d':
            outlineout = geoout
            outlines = outlineout.GetPolys()
            outline_data_vtk = outlines.GetData()

        else:
            raise KeyError(
                "The problem dimension must be either '2d' or '3d' "
                f"but you provided {self.problem_dimension}! Abort..."
            )

        outline_data = vtk_to_numpy(outline_data_vtk)
        return outlineout, outline_data

    def _visualize_intermediate_discrepancy_measure(
        self, points, vertices, intersectionpoints, point_vector
    ):
        """Visualize intermediate discrepancy measure in vtk."""
        if self.visualization_bool:
            for j in point_vector:
                x = points.InsertNextPoint(j)
                vertices.InsertNextCell(1)
                vertices.InsertCellPoint(x)

            for idx, j in enumerate(intersectionpoints):
                x = points.InsertNextPoint(j)
                vertices.InsertNextCell(idx + 1)
                vertices.InsertCellPoint(x)

    def _visualize_final_discrepancy_measure(self, outlineout, points, vertices):
        """Visualize the final discrepancy measure in vtk."""
        if self.visualization_bool:
            colors = vtk.vtkNamedColors()
            renderer = vtk.vtkRenderer()
            renWin = vtk.vtkRenderWindow()
            renWin.AddRenderer(renderer)
            iren = vtk.vtkRenderWindowInteractor()
            iren.SetRenderWindow(renWin)

            pointdata = vtk.vtkPolyData()

            pointdata.SetPoints(points)
            pointdata.SetVerts(vertices)

            ugridMapper = vtk.vtkDataSetMapper()
            ugridMapper.SetInputData(outlineout)

            pointMapper = vtk.vtkPolyDataMapper()

            pointMapper.SetInputData(pointdata)
            pointActor = vtk.vtkActor()
            pointActor.SetMapper(pointMapper)
            pointActor.GetProperty().SetColor([0.0, 0.0, 1.0])
            pointActor.GetProperty().SetPointSize(10)
            pointActor.GetProperty().SetRenderPointsAsSpheres(True)

            ugridActor = vtk.vtkActor()
            ugridActor.SetMapper(ugridMapper)
            ugridActor.GetProperty().SetColor(colors.GetColor3d("Peacock"))
            ugridActor.GetProperty().EdgeVisibilityOn()

            renderer.AddActor(ugridActor)
            renderer.AddActor(pointActor)
            renderer.SetBackground(colors.GetColor3d("Beige"))

            renderer.ResetCamera()
            renderer.GetActiveCamera().Elevation(1.0)
            renderer.GetActiveCamera().Azimuth(0.01)
            renderer.GetActiveCamera().Dolly(0)

            renWin.SetSize(640, 480)

            # Generate viewer
            renWin.Render()
            iren.Start()

    def _stretch_vector(self, vec1, vec2, scalar):
        """Extend a vector by scalar factor on both ends.

        Args:
            vec1 (list): root point coordinates
            vec2 (list): directional point coordinates
            scalar (float): scalar multiplier

        Returns:
            vec (list): vector from modified root to modified direction point
        """
        vec = [[], []]

        for vec1_ele, vec2_ele in zip(vec1, vec2):
            vec[0].append(vec1_ele - scalar * (vec2_ele - vec1_ele))
            vec[1].append(vec2_ele + scalar * (vec2_ele - vec1_ele))

        return vec

    def _compute_distance(self, intersection_points, measured_points):
        """Find the furthest point for a set of intersection points.

        Args:
            intersection_points (list): intersection point coordinates
            measured_points (list): pair of points from monitor file

        Returns:
            distance (float): signed distance between root point and furthest outward
                            intersection point; positive if in positive direction from root
        """
        distance = np.inf

        np1m = np.array(measured_points[0])
        np2m = np.array(measured_points[1])

        for p in intersection_points:
            npp = np.array(p)
            dist = np.linalg.norm(p - np1m, ord=2)
            if np.dot(np2m - np1m, npp - np1m) < 0:
                dist *= -1
            if dist < distance:
                distance = dist

        return distance

    def create_UnstructuredGridFromEnsight_per_time_step(self, time):
        """Read ensight file.

        Afterwards, warpbyvector by displacement of *structure*
        result in case files.

        Args:
            time (float): time value for data processing - executed once for every time value

        Returns:
            grid (vtkUnstructuredGrid): deformed discretization for given time
        """
        timestepsinensight = self.raw_file_data.GetTimeSets()

        timesiter = timestepsinensight.NewIterator()
        timesiter.GoToFirstItem()

        steps = np.array([])

        while not timesiter.IsDoneWithTraversal():
            curr = timesiter.GetCurrentObject()
            steps = np.append(steps, vtk_to_numpy(curr))
            timesiter.GoToNextItem()

        steps = np.unique(steps)
        idx = np.where(abs(steps - time) < self.time_tol)
        ensight_time = steps[idx]

        if len(ensight_time) > 1:
            raise ValueError(
                'point in time from *.monitor file used with time_tol is not unique in results'
            )
        elif len(ensight_time) == 0:
            raise ValueError(
                'point in time from *.monitor file used with time_tol not existing in results'
            )

        self.raw_file_data.SetTimeValue(ensight_time)
        self.raw_file_data.Update()

        readout = self.raw_file_data.GetOutput()
        number_of_blocks = readout.GetNumberOfBlocks()
        if number_of_blocks != 1:
            raise ValueError(
                'ensight reader output has more or less than one block. This is not expected.'
                'Investigate your data!'
            )
        block = readout.GetBlock(0)

        block.GetPointData().SetActiveVectors(self.displacement_fields[0])

        vtk_warp_vector = vtk.vtkWarpVector()
        vtk_warp_vector.SetScaleFactor(1.0)
        vtk_warp_vector.SetInputData(block)
        vtk_warp_vector.Update()
        if len(self.displacement_fields) > 1:
            for i, field in enumerate(self.displacement_fields):
                if i > 0:
                    secondblock = vtk_warp_vector.GetOutput()
                    secondblock.GetPointData().SetActiveVectors(field)
                    wvb = vtk.vtkWarpVector()
                    wvb.SetScaleFactor(1.0)
                    wvb.SetInputData(secondblock)
                    wvb.Update()
                    vtk_warp_vector = wvb

        grid = vtk_warp_vector.GetUnstructuredGridOutput()

        return grid
