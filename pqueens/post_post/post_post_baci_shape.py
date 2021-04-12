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
        self,
        path_ref_data,
        time_tol,
        visualization,
        delete_field_data,
        file_prefix,
        file_postfix,
        displacement_fields,
        problem_dimension,
    ):
        """Initialize post_post_baci_shape class

        Args:
            path_ref_data (string): experimental data in monitor format
            time_tol (float): time tolerance for given reference timepoints
            visualization (bool): boolean for vtk visualization control
            file_prefix (string): file prefix in result data to find result files
            file_postfix (string): file postfix and ending to find result files
            displacement_fields (list): strings with exact field names for displacement to apply
            problem_dimension (string): string to determine problems spatial dimension

        Returns:
            post_post (post_post object)
        """
        super(PostPostBACIShape, self).__init__(delete_field_data, file_prefix)
        self.path_ref_data = path_ref_data
        self.time_tol = time_tol
        self.visualizationon = visualization
        self.file_prefix = file_prefix
        self.file_postfix = file_postfix
        self.displacement_fields = displacement_fields
        self.problem_dimension = problem_dimension
        # this is ugly, but necessary as there is no other method only called once
        self.ref_data = self.read_monitorfile()

    @classmethod
    def from_config_create_post_post(cls, base_settings):
        """Create post_post routine from problem description

        Args:
            base_settings (list): input read from json input file

        Returns:
            post_post (post_post object)
        """
        post_post_options = base_settings['options']
        path_ref_data = post_post_options['path_to_ref_data']
        time_tol = post_post_options['time_tol']
        visualization = post_post_options.get('visualization', False)
        file_prefix = post_post_options['file_prefix']
        file_postfix = post_post_options.get('file_postfix', 'structure.case')
        displacement_fields = post_post_options.get('displacement_fields', ['displacement'])
        problem_dimension = post_post_options.get('problem_dimension', '2d')
        delete_field_data = post_post_options.get('delete_field_data', False)
        return cls(
            path_ref_data,
            time_tol,
            visualization,
            delete_field_data,
            file_prefix,
            file_postfix,
            displacement_fields,
            problem_dimension,
        )

    # ------------------------ COMPULSORY CHILDREN METHODS ------------------------
    def read_post_files(self, file_names, **kwargs):
        """
        main evaluation routine of post and post_post are located here
        residual vector has signed scalar entries for distances between surface compared to
        experiment surface defined via *.monitor file. *.monitor file is given in BACI heritage
        format, so it is reusable.

        write residual to self.result.

        Returns:
            None
        """

        path = glob.glob(file_names + '*' + self.file_postfix)

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

    def error_handling(self, output_dir):

        """
        error handling function

        Args:
            output_dir (string): location to copy failed input files to

        Returns:
            None

        """
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
        with open(self.path_ref_data) as mymonitorfile:
            lines = mymonitorfile.readlines()
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

    def create_mesh_and_intersect_vtk(self, path):

        """
        Create vtk representation of deformed external_geometry_obj and evaluate surface distance
        measurement for every given timestep from the experiment. (More than one for example for
        viscous behaviour)

        Args:
            ref_data (list): formatted list from monitor file
            path (string): path to .case result

        Returns:
            residual (list): full residual from this post_post class

        """
        residual = []
        # get visual feedback for the intersection problems
        self.visualizationon
        for meas_iter_t, measurementCurrStep in enumerate(self.ref_data):

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

            grid = self.create_UnstructuredGridFromEnsight(path, measurementCurrStep[0])

            geo = vtk.vtkGeometryFilter()
            geo.SetInputData(grid)
            geo.Update()
            geoout = geo.GetOutput()

            # switch between problem dimension as vtk functions depend on it
            if self.problem_dimension == '2d':
                # following part is 2D specific
                outline = vtk.vtkFeatureEdges()
                outline.SetInputData(geoout)
                outline.Update()
                outlineout = outline.GetOutput()

                outlines = outlineout.GetLines()
                odata = outlines.GetData()

            elif self.problem_dimension == '3d':
                outlineout = geoout
                outlines = outlineout.GetPolys()
                odata = outlines.GetData()

            npodata = vtk_to_numpy(odata)

            for meas_iter, measuredPointPair in enumerate(measurementCurrStep[1]):
                vec = self.stretch_vector(measuredPointPair[0], measuredPointPair[1], 10)

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

                distance = self.compute_distance(intersectionpoints, measuredPointPair)

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

        block.GetPointData().SetActiveVectors(self.displacement_fields[0])

        wv = vtk.vtkWarpVector()
        wv.SetScaleFactor(1.0)
        wv.SetInputData(block)
        wv.Update()
        if len(self.displacement_fields) > 1:
            for i, field in enumerate(self.displacement_fields):
                if i > 0:
                    secondblock = wv.GetOutput()
                    secondblock.GetPointData().SetActiveVectors(field)
                    wvb = vtk.vtkWarpVector()
                    wvb.SetScaleFactor(1.0)
                    wvb.SetInputData(secondblock)
                    wvb.Update()
                    wv = wvb

        grid = wv.GetUnstructuredGridOutput()

        return grid
