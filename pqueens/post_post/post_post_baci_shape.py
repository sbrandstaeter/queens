import glob
import os
import numpy as np
import re
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from pqueens.utils.run_subprocess import run_subprocess

from .post_post import PostPost


class PostPostBACIShape(PostPost):
    """
    class for distance to surface measurement post_post evaluation
    this post_post class uses full ensight result in vtk to measure distance of surface from
    simulations to experiment
    """

    def __init__(
        self, path_ref_data, time_tol, case_type, visualization, delete_field_data, file_prefix
    ):

        super(PostPostBACIShape, self).__init__(delete_field_data, file_prefix)
        self.path_ref_data = path_ref_data
        self.time_tol = time_tol
        self.case_type = case_type
        self.visualizationon = visualization
        self.file_prefix = file_prefix

        # this is ugly, but necessary as there is no other method only called once
        self.ref_data = self.read_monitorfile()

    @classmethod
    def from_config_create_post_post(cls, base_settings):
        """ Create post_post routine from problem description

        Args:
            config: input json file with problem description

        Returns:
            post_post (post_post object)
        """
        post_post_options = base_settings['options']
        path_ref_data = post_post_options['path_to_ref_data']
        time_tol = post_post_options['time_tol']
        case_type = post_post_options['case_type']
        visualization = post_post_options['visualization']
        file_prefix = post_post_options['file_prefix']
        delete_field_data = post_post_options['delete_field_data']
        return cls(
            path_ref_data, time_tol, case_type, visualization, delete_field_data, file_prefix
        )

    # ------------------------ COMPULSORY CHILDREN METHODS ------------------------
    def read_post_files(self):
        """
        main evaluation routine of post and post_post are located here
        residual vector has signed scalar entries for distances between surface compared to
        experiment surface defined via *.monitor file. *.monitor file is given in BACI heritage
        format, so it is reusable.

        write residual to self.result.

        Returns:
            None
        """

        prefix_expr = '*' + self.file_prefix + '*'
        files_of_interest = os.path.join(self.output_dir, prefix_expr)

        # we can make use of specific cut output here
        if self.case_type == 'cut_fsi':
            path = glob.glob(files_of_interest + 'boundary_of_structure' + '*' + '.case')
        elif self.case_type in ['2d_full', '3d_full']:
            path = glob.glob(files_of_interest + '*' + 'structure.case')
        else:
            raise ValueError('case_type unknown')

        # glob returns arbitrary list -> need to sort the list before using
        path.sort()

        if len(path) != 1:
            raise ValueError('We need exactly one *.case result.')

        post_out = self.create_mesh_and_intersect_vtk(path[0])

        self.error = False
        self.result = post_out

    def delete_field_data(self):
        """
        Delete output files except files with given prefix

        Returns:
             None

        """

        inverse_prefix_expr = r"[!" + self.file_prefix + r"]*"
        files_of_interest = os.path.join(self.output_dir, inverse_prefix_expr)
        post_file_list = glob.glob(files_of_interest)
        for filename in post_file_list:
            command_string = "rm " + filename
            # "cd " + self.output_file + "&& ls | grep -v --include=*.{mon,csv} | xargs rm"
            _, _, _, _ = run_subprocess(command_string)

    def error_handling(self):
        # TODO  ### Error Types ###
        # No QoI file
        # Time/Time step not reached
        # Unexpected values

        # Organized failed files
        input_file_extention = 'dat'
        if self.error is True:
            command_string = (
                "cd "
                + self.output_dir
                + "&& cd ../.. && mkdir -p postpost_error && cd "
                + self.output_dir
                + r"&& cd .. && mv *."
                + input_file_extention
                + r" ../postpost_error/"
            )
            _, _, _, _ = run_subprocess(command_string)

    def read_monitorfile(self):
        """
        Read Monitor file
        The Monitor File contains measurements from the experiments. It is formatted like for BACI.

        Returns:
            monfile_data (list): data from monitor file in numbers
        """
        monitorcontent = open(self.path_ref_data).readlines()
        i = 0
        npoints = 0
        steps = 0
        npoint_lines = []
        steps_lines = []
        for line in monitorcontent:
            if line.startswith('#'):
                continue
            if line.startswith('steps'):
                firstline = line
                steps = re.findall('^(?:steps )(.+)(?= npoints)', firstline, re.M)
                steps = int(steps[0])
                npoints = re.findall('^(?:steps )(?:.+)?(?: npoints )(.+)', firstline, re.M)
                npoints = int(npoints[0])
                continue
            if i < npoints:
                npoint_lines.append(line)
                i += 1
                continue
            if i - npoints - 1 < steps:
                steps_lines.append(line)

        if npoints == 0 or steps == 0:
            raise ValueError(
                'read_monitorfile did not find useful content. Monitor format is ' 'probably wrong'
            )

        npoint_lines = [i.strip() for i in npoint_lines]
        npoint_lines = [i.split() for i in npoint_lines]
        npoint_lines = [[int(ii) for ii in i] for i in npoint_lines]

        steps_lines = [i.split() for i in steps_lines]
        steps_lines = [[float(ii) for ii in i] for i in steps_lines]

        monfile_data = []
        for i in steps_lines:
            monfile_data.append(
                [[0.0e0], [[[0, 0, 0] for j in range(0, 2)] for k in range(0, npoints)]]
            )

        for i in range(len(steps_lines)):
            k = 1
            monfile_data[i][0] = steps_lines[i][0]
            for ii in range(len(npoint_lines)):
                for x in range(0, 2):
                    for iii in range(0, npoint_lines[ii][0]):
                        monfile_data[i][1][ii][x][npoint_lines[ii][iii + 1]] = steps_lines[i][k]
                        k += 1

        return monfile_data

    def create_mesh_and_intersect_vtk(self, path):

        """
        Create vtk representation of deformed geometry and evaluate surface distance measurement for
        every given timestep from the experiment. (More than one for example for viscous behaviour)
        Args:
            ref_data (list): formatted list from monitor file
            path (string): path to .case result
        Returns:
            residual (list): full residual from this post_post class
        """
        residual = []
        # get visual feedback for the intersection problems
        self.visualizationon
        for meas_iter_t, meas_t in enumerate(self.ref_data):

            if self.visualizationon:
                colors = vtk.vtkNamedColors()
                renderer = vtk.vtkRenderer()
                renWin = vtk.vtkRenderWindow()
                renWin.AddRenderer(renderer)
                iren = vtk.vtkRenderWindowInteractor()
                iren.SetRenderWindow(renWin)

                pointdata = vtk.vtkPolyData()
                points = vtk.vtkPoints()
                vertices = vtk.vtkCellArray()

            grid = self.create_UnstructuredGridFromEnsight(path, meas_t[0])

            geo = vtk.vtkGeometryFilter()
            geo.SetInputData(grid)
            geo.Update()
            geoout = geo.GetOutput()

            if self.case_type == '2d_full':
                # following part is 2D specific
                outline = vtk.vtkFeatureEdges()
                outline.SetInputData(geoout)
                outline.Update()
                outlineout = outline.GetOutput()

                outlines = outlineout.GetLines()
                odata = outlines.GetData()

            elif self.case_type in ['cut_fsi', '3d_full']:
                outlineout = geoout
                outlines = outlineout.GetPolys()
                odata = outlines.GetData()

            npodata = vtk_to_numpy(odata)

            for meas_iter, meas in enumerate(meas_t[1]):
                vec = self.stretch_vector(meas[0], meas[1], 10)

                i = 0

                intersectionpoints = []

                while i < len(npodata):
                    numpoints = npodata.item(i)
                    locpointids = npodata[i + 1 : i + 1 + numpoints]

                    locations = []
                    for id in np.nditer(locpointids):
                        x = [0, 0, 0]
                        outlineout.GetPoint(id, x)
                        locations.append(x)

                    local_points = vtk.vtkPoints()
                    for p in range(0, len(locations)):
                        local_points.InsertNextPoint(locations[p])

                    if len(locations) == 2:
                        local_element = vtk.vtkLine()
                    elif len(locations) == 3:
                        local_element = vtk.vtkTriangle()
                    elif len(locations) == 4:
                        local_element = vtk.vtkQuad()
                    else:
                        raise ValueError(
                            'Unknown local_element type for structure surface discretization.'
                        )

                    local_element.Initialize(len(locations), local_points)

                    intersectionpoint = [0, 0, 0]
                    pcoords = [0, 0, 0]
                    t = 0
                    uselessint = 0
                    # key line of the algorithm: line intersection
                    intersectionfound = local_element.IntersectWithLine(
                        vec[0],
                        vec[1],
                        1e-12,
                        vtk.reference(t),
                        intersectionpoint,
                        pcoords,
                        vtk.reference(uselessint),
                    )

                    if intersectionfound:
                        intersectionpoints.append(intersectionpoint)

                    i += numpoints
                    i += 1

                distance = self.compute_distance(intersectionpoints, meas)

                residual.append(distance)

                if self.visualizationon:
                    for j in vec:
                        x = points.InsertNextPoint(j)
                        vertices.InsertNextCell(1)
                        vertices.InsertCellPoint(x)

                    for id, j in enumerate(intersectionpoints):
                        x = points.InsertNextPoint(j)
                        vertices.InsertNextCell(id + 1)
                        vertices.InsertCellPoint(x)

            if self.visualizationon:
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
                # pointActor.GetProperty().VertexVisibilityOff()

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

                # Generate viewer.
                renWin.Render()
                iren.Start()

        return residual

    def stretch_vector(self, vec1, vec2, scalar):
        """
        short utility function to extend a vector by scalar factor on both ends.

        Args:
            vec1 (list): root point coordinates
            vec2 (list): directional point coordinates
            scalar (float): scalar multiplier

        Returns:
            vec (list): vector from modified root to modified direction point
        """

        vec = [[], []]

        for i, itervec in enumerate(vec1):
            vec[0].append(vec1[i] - scalar * (vec2[i] - vec1[i]))
            vec[1].append(vec2[i] + scalar * (vec2[i] - vec1[i]))

        return vec

    def compute_distance(self, intersection_points, measured_points):
        """
        short utility function to find the furthest outwards laying point for a set of
        intersection points

        Args:
            intersection_points (list): intersection point coordinates
            measured_points (list): pair of points from monitor file

        Returns:
            distance (float): signed distance between root point and furthest outward
                            intersection point; positive if in positive direction from root
        """

        distance = float('inf')

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

    def create_UnstructuredGridFromEnsight(self, path, time):

        """
        read ensight file from post_processor and warpbyvector by displacement of *structure* result
        in case files

        Args:
            path (string): experiment directory
            time (float): time value for post_post process - executed once for every time value
        Returns:
            grid (vtkUnstructuredGrid): deformed discretization for given time
        """

        reader = vtk.vtkGenericEnSightReader()
        reader.SetCaseFileName(path)
        reader.ReadAllVariablesOn()
        reader.Update()

        timestepsinensight = reader.GetTimeSets()

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

        reader.SetTimeValue(ensight_time)
        reader.Update()

        readout = reader.GetOutput()
        numblo = readout.GetNumberOfBlocks()
        if numblo != 1:
            raise ValueError(
                'ensightreaderoutput has more or less than one block. This is not expected.'
                'Investigate your data!'
            )
        block = readout.GetBlock(0)

        if self.case_type in ['2d_full', '3d_full']:
            block.GetPointData().SetActiveVectors('displacement')
        elif self.case_type in ['cut_fsi']:
            block.GetPointData().SetActiveVectors('idispnp')

        wv = vtk.vtkWarpVector()
        wv.SetScaleFactor(1.0)
        wv.SetInputData(block)
        wv.Update()
        grid = wv.GetUnstructuredGridOutput()

        return grid
