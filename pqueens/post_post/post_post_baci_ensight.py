import glob

import numpy as np
import pandas as pd
import vtk
from pqueens.database.mongodb import MongoDB
from pqueens.external_geometry.external_geometry import ExternalGeometry
from pqueens.post_post.post_post import PostPost
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy


class PostPostBACIEnsight(PostPost):
    """ Class for post-post-processing BACI Ensight output

    Attributes:
        field_specification_dict (dict): Dictionary containing description of the output
                                            fields which should be read in
        db (obj): Database object
        experiment_name (str): Name of the current QUEENS experiment
        external_geometry_obj (obj): QUEENS external geometry object
        experimental_data (pd.DataFrame): Pandas dataframe with experimental data
        coordinates_label_experimental (lst): List of (spatial) coordinate labels
                                                of the experimental data set
        output_label_experimental (str): Output label of the experimental data set
        time_label_experimental (str): Time label of the experimental data set

    """

    def __init__(
        self,
        delete_data_flag,
        post_post_file_name_prefix_lst,
        field_specification_dict,
        database,
        experiment_name,
        experimental_data,
        coordinates_label_experimental,
        output_label_experimental,
        time_label_experimental,
        external_geometry_obj,
    ):
        """
        Init method for ensight post-post object.

        Args:
            delete_data_flag (bool):  Delete files after processing
            post_post_file_name_prefix_lst (lst): List with prefixes of result files
            field_specification_dict (dict): Dictionary containing description of the output
                                             fields which should be read in
            db (obj): Database object
            experiment_name (str): Name of QUEENS experiment
            experimental_data (pd.DataFrame): Pandas dataframe with experimental data
            coordinates_label_experimental (lst): List of (spatial) coordinate labels
                                                  of the experimental data set
            output_label_experimental (str): Output label of the experimental data set
            time_label_experimental (str): Time label of the experimental data set
            external_geometry_obj (obj): QUEENS external geometry object

        Returns:
            Instance of PostPostBACIEnsight class (obj)

        """
        super(PostPostBACIEnsight, self).__init__(delete_data_flag, post_post_file_name_prefix_lst)
        self.field_specification_dict = field_specification_dict
        self.db = database
        self.experiment_name = experiment_name
        self.experimental_data = experimental_data
        self.coordinates_label_experimental = coordinates_label_experimental
        self.ouput_label_experimental = output_label_experimental
        self.time_label_experimental = time_label_experimental
        self.external_geometry_obj = external_geometry_obj

    @classmethod
    def from_config_create_post_post(cls, options):
        """ Create post_post routine from problem description

        Args:
            options (dict): input options

        Returns:
            post_post: PostPostBACIEnsight object

        """
        post_post_options = options['options']
        delete_data_flag = post_post_options['delete_field_data']
        post_post_file_name_prefix_lst = post_post_options['post_post_file_name_prefix_lst']
        assert isinstance(
            post_post_file_name_prefix_lst, list
        ), "The option post_post_file_name_prefix_lst must be of type list!"

        field_specification_dict = post_post_options['field_specification']

        len_time = len(field_specification_dict['target_time'])
        len_field_type = len(field_specification_dict['physical_field_dict']['vtk_field_type'])
        len_array_type = len(field_specification_dict['physical_field_dict']['vtk_array_type'])
        len_field_label = len(field_specification_dict['physical_field_dict']['vtk_field_label'])
        len_field_comp = len(field_specification_dict['physical_field_dict']['field_components'])

        if not len_time == len_field_type == len_array_type == len_field_label == len_field_comp:
            raise IOError("List length of ensight field specifications have to be the same!")

        database = MongoDB.from_config_create_database(options['config'])

        # get some properties of the experimental data
        options['config']['database']['drop_all_existing_dbs'] = False
        experiment_name = options['config']['global_settings']['experiment_name']
        experimental_data = database.load(experiment_name, 1, 'experimental_data')
        experimental_data = pd.DataFrame.from_dict(experimental_data)

        model_name = options['config']['method']['method_options']['model']
        coordinates_label_experimental = options['config'][model_name].get('coordinate_labels')
        ouput_label_experimental = options['config'][model_name].get('output_label')
        time_label_experimental = options['config'][model_name].get('time_label')

        external_geometry_obj = ExternalGeometry.from_config_create_external_geometry(
            options['config']
        )

        return cls(
            delete_data_flag,
            post_post_file_name_prefix_lst,
            field_specification_dict,
            database,
            experiment_name,
            experimental_data,
            coordinates_label_experimental,
            ouput_label_experimental,
            time_label_experimental,
            external_geometry_obj,
        )

    def read_post_files(self, files_of_interest, **kwargs):
        """
        Read-in EnSight files and interpolate fields to specified coordinates (e.g., of the
        underlying experimental data).
        Return then only the interpolated field values at these locations in form of an array.

        Args:
            files_of_interest (lst): List of file paths that should be read-in
            kwargs (dict): Further optional arguments as key-value pairs

        Returns:
            None

        """
        index = kwargs.get('idx')
        try:
            time_tol = self.field_specification_dict['time_tol'][index]
        except KeyError:
            time_tol = None

        target_time = self.field_specification_dict['target_time'][index]
        vtk_field_type = self.field_specification_dict['physical_field_dict']['vtk_field_type'][
            index
        ]
        vtk_array_type = self.field_specification_dict['physical_field_dict']['vtk_array_type'][
            index
        ]
        vtk_field_label = self.field_specification_dict['physical_field_dict']['vtk_field_label'][
            index
        ]
        vtk_field_components = self.field_specification_dict['physical_field_dict'][
            'field_components'
        ][index]
        geometric_target = self.field_specification_dict["geometric_target"][index]

        # Set vtk reader object
        vtk_reader_obj = vtk.vtkEnSightGoldBinaryReader()

        post_file = glob.glob(files_of_interest + "_" + vtk_field_type + ".case")
        post_file.sort()
        # glob returns arbitrary list -> need to sort the list before using
        if len(post_file) > 1:
            raise IOError("There can only be one .case file per BACI field")

        # determine the correct time vector depending on input specification
        if target_time[0] == "from_experimental_data":
            # get unique time list
            time_vec = list(set(self.experimental_data[self.time_label_experimental]))
        else:
            time_vec = target_time

        # loop over differnt time-steps here.
        time_vec = sorted(time_vec)
        for time_value in time_vec:
            try:
                # here we load the EnSight Files (only one post file should be found)
                vtk_reader_obj.SetCaseFileName(post_file[0])  # index 0 as only one file
                vtk_reader_obj.Update()
                post_data = self._vtk_from_ensight(vtk_reader_obj, time_value, time_tol)

                # check if data was found
                if not post_data:
                    self.error = True
                    self.result = None
                    break

            except FileNotFoundError:
                self.error = True
                self.result = None
                break

            # get ensight field from specified coordinates
            result = self._get_field_values_by_coordinates(
                post_data,
                vtk_array_type,
                vtk_field_label,
                vtk_field_components,
                geometric_target,
                time_value,
            )

            # potentially append new time step result to result array
            if self.result is not None:
                self.result = np.hstack((self.result, result))
            else:
                self.result = result

    def _get_field_values_by_coordinates(
        self,
        vtk_post_data_obj,
        vtk_array_type,
        vtk_field_label,
        vtk_field_components,
        geometric_target,
        time_value,
    ):
        """
        Interpolate the vtk field to the coordinates from experimental data

        Args:
            vtk_post_data_obj (obj): VTK ensight object that contains that solution fields of
                                     interest
            vtk_field_label (str): Field label that should be extracted
            vtk_field_components (lst): List of components of the respective fields that should be
                                        extracted
            vtk_array_type (str): Type of vtk array (cell array or point array)

            geometric_target (lst): List specifying where (for which geometric target) vtk data
                                    should be read-in. The first list element can have the entries:

                                    - "geometric_set": Data is read-in from a specific geometric
                                                       set
                                                       in BACI
                                    - "experimental_data": Data is read-in at experimental data
                                                           coordinates (which must be part of the
                                                           domain)

                                    The second list entry specifies the label or name of the
                                    geometric set or the experimental data.
            time_value (float): Current time value at which simulation should be evaluated

        Returns:
            interpolated_data (np.array): Array of field values at specified coordinates/geometric
                                          sets

        """
        if geometric_target[0] == "experimental_data":
            response_data = self._get_data_from_experimental_coordinates(
                vtk_post_data_obj, vtk_array_type, vtk_field_label, vtk_field_components, time_value
            )
        elif geometric_target[0] == "geometric_set":
            response_data = self._get_data_from_geometric_set(
                vtk_post_data_obj,
                vtk_array_type,
                vtk_field_label,
                vtk_field_components,
                geometric_target[1],
            )
        else:
            raise ValueError(
                "Geometric target for ensight vtk must be eiter 'geometric_set' or"
                f"'experimental_data'. You provided '{geometric_target[0]}'. Abort..."
            )

        return response_data

    def _get_data_from_experimental_coordinates(
        self, vtk_post_data_obj, vtk_array_type, vtk_field_label, vtk_field_components, time_value
    ):
        """
        Interpolate the ensight/vtk field to experimental coordinates

        Args:
            vtk_post_data_obj (obj): VTK ensight object that contains that solution fields of
                                     interest
            vtk_array_type (str): Type of vtk array (cell array or point array)
            vtk_field_label (str): Field label that should be extracted
            vtk_field_components (lst): List of components of the respective fields that should
                                        be extracted
            time_value (float): Current time value at which the simulation shall be evaluated

        Returns:
            interpolated_data (np.array): Array of field values interpolated to coordinates
                                          of experimental data

        """
        experimental_coordinates_for_snapshot = []
        if self.time_label_experimental:
            for _, row in self.experimental_data.iterrows():
                if time_value == row[self.time_label_experimental]:
                    experimental_coordinates_for_snapshot.append(
                        row[self.coordinates_label_experimental].to_list()
                    )
        else:
            for coord in self.coordinates_label_experimental:
                experimental_coordinates_for_snapshot.append(
                    np.array(self.experimental_data[coord]).reshape(-1, 1)
                )
            if len(experimental_coordinates_for_snapshot) != 3:
                raise ValueError(f"Please provide 3d coordinates in the observation data")
            experimental_coordinates_for_snapshot = np.concatenate(
                experimental_coordinates_for_snapshot, axis=1
            )
        # interpolate vtk solution to experimental coordinates
        interpolated_data = PostPostBACIEnsight._interpolate_vtk(
            experimental_coordinates_for_snapshot,
            vtk_post_data_obj,
            vtk_array_type,
            vtk_field_label,
            vtk_field_components,
        )

        return interpolated_data

    def _get_data_from_geometric_set(
        self,
        vtk_post_data_obj,
        vtk_array_type,
        vtk_field_label,
        vtk_field_components,
        geometric_set,
    ):
        """
        Get entire fields for desired components at all nodes of desired geometric set.

        Args:
            vtk_post_data_obj (obj): VTK ensight object that contains that solution fields of
                                     interest
            vtk_array_type (str): Type of vtk array (cell array or point array)
            vtk_field_label (str): Field label that should be extracted
            vtk_field_components (lst): List of components of the respective fields that should
                                        be extracted
            geometric_set (str): Label/name of geometric set.

        Returns:
            data (np.array): Array of field values for nodes of geometric set

        """
        geometric_set_data = self.db.load(self.experiment_name, 1, 'geometric_sets')
        if geometric_set_data is None:
            self.write_geometry_coordinates_to_db()
            geometric_set_data = self.db.load(self.experiment_name, 1, 'geometric_sets')

        # get node coordinates by geometric set, loop over all topologies
        for nodes in geometric_set_data['node_topology']:
            if nodes['topology_name'] == geometric_set:
                nodes_of_interest = nodes['node_mesh']

        for lines in geometric_set_data['line_topology']:
            if lines['topology_name'] == geometric_set:
                nodes_of_interest = lines['node_mesh']

        for surfs in geometric_set_data['surface_topology']:
            if surfs['topology_name'] == geometric_set:
                nodes_of_interest = surfs['node_mesh']

        for vols in geometric_set_data['volume_topology']:
            if vols['topology_name'] == geometric_set:
                nodes_of_interest = vols['node_mesh']

        both = set(nodes_of_interest).intersection(
            geometric_set_data['node_coordinates']['node_mesh']
        )
        indices = [geometric_set_data['node_coordinates']['node_mesh'].index(x) for x in both]
        geometric_set_coordinates = [
            geometric_set_data['node_coordinates']['coordinates'][index] for index in indices
        ]

        # interpolate vtk solution to experimental coordinates
        interpolated_data = PostPostBACIEnsight._interpolate_vtk(
            geometric_set_coordinates,
            vtk_post_data_obj,
            vtk_array_type,
            vtk_field_label,
            vtk_field_components,
        )

        return interpolated_data

    @staticmethod
    def _interpolate_vtk(
        coordinates, vtk_post_data_obj, vtk_array_type, vtk_field_label, vtk_field_components
    ):
        """
        Interpolate the vtk solution field to given coordinates.

        Args:
            coordinates (np.array): Coordinates at which the vtk solution field should be
                                    interpolated at
            vtk_post_data_obj (obj): VTK ensight object that contains that solution fields of
                                     interest
            vtk_array_type (str): Type of vtk array (cell array or point array)
            vtk_field_label (str): Field label that should be extracted
            vtk_field_components (lst): List of components of the respective fields that should
                                        be extracted

        Returns:
            interpolated_data (np.array): Numpy array with solution data interpolated to
                                          respective coordinates.

        """

        # -- create vtk point object from experimental coordinates --
        vtk_points_obj = vtk.vtkPoints()
        vtk_points_obj.SetData(numpy_to_vtk(coordinates))
        # from vtk point obj create polydata obj
        polydata_obj = vtk.vtkPolyData()
        polydata_obj.SetPoints(vtk_points_obj)

        # Generate a vtk filter with desired vtk data as source
        probe_filter_obj = vtk.vtkProbeFilter()
        probe_filter_obj.SetSourceData(vtk_post_data_obj)

        # Evalutate/interpolate this filter at experimental coordinates
        probe_filter_obj.SetInputData(polydata_obj)
        probe_filter_obj.Update()

        # Return the fields as numpy array
        if vtk_array_type == 'point_array':
            field = vtk_to_numpy(
                probe_filter_obj.GetOutput().GetPointData().GetArray(vtk_field_label)
            )

        elif vtk_array_type == 'cell_array':
            field = vtk_to_numpy(
                probe_filter_obj.GetOutput().GetCellData().GetArray(vtk_field_label)
            )

        else:
            raise ValueError(
                "VTK array type must be either 'point_array' or 'cell_array', but you provided"
                f"{vtk_array_type}! Abort..."
            )

        data = []

        for i in vtk_field_components:
            if field.ndim == 1:
                data.append(field.reshape(-1, 1))
            else:
                data.append(field[:, i].reshape(-1, 1))
        data = np.concatenate(data, axis=1)

        # QUEENS expects a float64 numpy object as result
        interpolated_data = data.astype('float64')

        return interpolated_data

    def _vtk_from_ensight(self, vtk_reader_obj, target_time, time_tol):
        """
        Load a vtk-object from the ensight BACI file.

        Args:
            vtk_reader_obj (obj): VTK reader object that contains data of interest
            target_time (float): Time the field should be evaluated on
            time_tol (float): Tolerance for the target time

        Retruns:
            vtk_solution_field (obj)

        """
        # Ensight contains different "timesets" which are containers for the actual data
        number_of_timesets = vtk_reader_obj.GetTimeSets().GetNumberOfItems()

        for num in range(number_of_timesets):
            time_set = vtk_to_numpy(vtk_reader_obj.GetTimeSets().GetItem(num))
            # Find the timeset that has more than one entry as sometimes there is an empty dummy
            # timeset in the ensight file (seems to be an artifact)
            if time_set.size > 1:
                # if the keyword `last` was provided, get the last timestep
                if target_time == 'last':
                    vtk_reader_obj.SetTimeValue(time_set[-1])
                else:
                    timestep = time_set.flat[np.abs(time_set - target_time).argmin()]

                    if np.abs(timestep - target_time) > time_tol:
                        raise RuntimeError(
                            "Time not within tolerance"
                            f"Target time: {target_time}, selected time: {timestep},"
                            f"tolerance {time_tol}"
                        )
                    vtk_reader_obj.SetTimeValue(timestep)

                vtk_reader_obj.Update()
                input_vtk = vtk_reader_obj.GetOutput()

                # the data is in the first block of the vtk object
                vtk_solution_field = input_vtk.GetBlock(0)

        return vtk_solution_field

    def write_geometry_coordinates_to_db(self):
        """
        Write geometry of interest to the database using QUEENS external geometry module.

        Returns:
            None

        """
        # read in the external geometry
        self.external_geometry_obj.main_run()

        geometric_set_dict = {
            "node_topology": self.external_geometry_obj.node_topology,
            "line_topology": self.external_geometry_obj.line_topology,
            "surface_topology": self.external_geometry_obj.surface_topology,
            "volume_topology": self.external_geometry_obj.volume_topology,
            "node_coordinates": self.external_geometry_obj.node_coordinates,
            "element_centers": self.external_geometry_obj.element_centers,
            "element_topology": self.external_geometry_obj.element_topology,
        }

        self.db.save(geometric_set_dict, self.experiment_name, 'geometric_sets', 1)
