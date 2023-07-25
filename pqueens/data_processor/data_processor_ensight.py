"""Data processor class for reading vtk-ensight data."""

import logging

import numpy as np
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from pqueens.data_processor.data_processor import DataProcessor
from pqueens.external_geometry import from_config_create_external_geometry
from pqueens.utils.experimental_data_reader import ExperimentalDataReader

_logger = logging.getLogger(__name__)


class DataProcessorEnsight(DataProcessor):
    """Class for data-processing ensight output.

    Attributes:
        experimental_data (dict): dict with experimental data.
        coordinates_label_experimental (lst): List of (spatial) coordinate labels
                                                of the experimental data set.
        time_label_experimental (str): Time label of the experimental data set.
        external_geometry_obj (obj): QUEENS external geometry object.
        target_time_lst (lst): Target time list for the ensight data, meaning time for which the
                                simulation state should be extracted.
        time_tol (float): Tolerance for the target time extraction.
        vtk_field_label (str): Label defining which physical field should be extraced from
                               the vtk data.
        vtk_field_components (lst): List with vector components that should be extracted
                                    from the field.
        vtk_array_type (str): Type of vtk array (e.g. *point_array*).
        geometric_target (lst): List with information about specific geometric target in vtk data
                                (This might be dependent on the simulation software that generated
                                the vtk file).
        geometric_set_data (dict): Dictionary describing the topology of the geometry
    """

    def __init__(
        self,
        data_processor_name,
        file_name_identifier=None,
        file_options_dict=None,
        files_to_be_deleted_regex_lst=None,
        external_geometry_obj=None,
        experimental_data_reader=None,
    ):
        """Init method for ensight data processor object.

        Args:
            data_processor_name (str): Name of the data processor.
            file_name_identifier (str): Identifier of file name.
                                             The file prefix can contain regex expression
                                             and subdirectories.
            file_options_dict (dict): Dictionary with read-in options for the file:
                - target_time_lst (lst): Target time list for the ensight data, meaning time for
                                         which the simulation state should be extracted
                - time_tol (float): Tolerance for the target time extraction
                - geometric_target (lst): List with information about specific geometric target in
                                          vtk data (This might be dependent on the simulation
                                          software that generated the vtk file.)
                - physical_field_dict (dict): Dictionary with options for physical field:
                    -- vtk_field_label (str): Label defining which physical field should be
                                              extracted from the vtk data
                    -- field_components (lst): List with vector components that should be extracted
                                        from the field
                    -- vtk_array_type (str): Type of vtk array (e.g., point_array)


            files_to_be_deleted_regex_lst (lst): List with paths to files that should be deleted.
                                                 The paths can contain regex expressions.
            external_geometry_obj (obj): QUEENS external geometry object
            experimental_data_reader (obj): Experimental data reader object

        Returns:
            Instance of DataProcessorEnsight class (obj)
        """
        super().__init__(
            data_processor_name=data_processor_name,
            file_name_identifier=file_name_identifier,
            file_options_dict=file_options_dict,
            files_to_be_deleted_regex_lst=files_to_be_deleted_regex_lst,
        )

        self._check_field_specification_dict(file_options_dict)
        # load the necessary options from the file options dictionary
        target_time_lst = file_options_dict.get('target_time_lst')
        if not isinstance(target_time_lst, list):
            raise TypeError(
                "The option 'target_time_lst' in the data_processor settings must be of type 'list'"
                f" but you provided type {type(target_time_lst)}. Abort..."
            )
        time_tol = file_options_dict.get('time_tol')
        if time_tol:
            if not isinstance(time_tol, float):
                raise TypeError(
                    "The option 'time_tol' in the data_processor settings must be of type 'float' "
                    f"but you provided type {type(time_tol)}. Abort..."
                )

        vtk_field_label = file_options_dict['physical_field_dict']['vtk_field_label']
        if not isinstance(vtk_field_label, str):
            raise TypeError(
                "The option 'vtk_field_label' in the data_processor settings must be of type 'str' "
                f"but you provided type {type(vtk_field_label)}. Abort..."
            )
        vtk_field_components = file_options_dict['physical_field_dict']['field_components']
        if not isinstance(vtk_field_components, list):
            raise TypeError(
                "The option 'field_components' in the data_processor settings must be of type "
                f"'list' but you provided type {type(vtk_field_components)}. Abort..."
            )

        vtk_array_type = file_options_dict['physical_field_dict']['vtk_array_type']
        if not isinstance(vtk_array_type, str):
            raise TypeError(
                "The option 'vtk_array_type' in the data_processor settings must be of type 'str' "
                f"but you provided type {type(vtk_array_type)}. Abort..."
            )

        # geometric_target (lst): List specifying where (for which geometric target) vtk data
        #                         should be read-in. The first list element can have the entries:
        #                         - "geometric_set": Data is read-in from a specific geometric set
        #                         - "experimental_data": Data is read-in at experimental data
        #                                  coordinates (which must be part of the domain)
        geometric_target = file_options_dict["geometric_target"]
        if not isinstance(geometric_target, list):
            raise TypeError(
                "The option 'geometric_target' in the data_processor settings must be of type "
                f"'list' but you provided type {type(geometric_target)}. Abort..."
            )

        self.geometric_set_data = None

        if geometric_target[0] == "geometric_set":
            self.geometric_set_data = self.read_geometry_coordinates(external_geometry_obj)

        if experimental_data_reader is not None:
            (
                _,
                _,
                _,
                self.experimental_data,
                self.time_label_experimental,
                self.coordinates_label_experimental,
            ) = experimental_data_reader.get_experimental_data()
        else:
            self.experimental_data = None
            self.time_label_experimental = None
            self.coordinates_label_experimental = None

        self.external_geometry_obj = external_geometry_obj
        self.target_time_lst = target_time_lst
        self.time_tol = time_tol
        self.vtk_field_label = vtk_field_label
        self.vtk_field_components = vtk_field_components
        self.vtk_array_type = vtk_array_type
        self.geometric_target = geometric_target

    @classmethod
    def from_config_create_data_processor(cls, config, data_processor_name):
        """Create ensight data processor from the problem description.

         Args:
             config (dict): Dictionary with problem description
             data_processor_name (str): Name of the data processor

        Returns:
             DataProcessorEnsight: Instance of DataProcessorEnsight
        """
        data_processor_options = config[data_processor_name].copy()
        data_processor_options.pop('type')

        # get experimental data to search for corresponding vtk data
        experimental_data_reader_name = data_processor_options.get("experimental_data_reader_name")
        experimental_data_reader = None
        if experimental_data_reader_name:
            experimental_data_reader = (
                ExperimentalDataReader.from_config_create_experimental_data_reader(
                    config, experimental_data_reader_name
                )
            )

        # generate the external geometry module
        external_geometry_obj = from_config_create_external_geometry(config, 'external_geometry')

        return cls(
            data_processor_name=data_processor_name,
            experimental_data_reader=experimental_data_reader,
            external_geometry_obj=external_geometry_obj,
            **data_processor_options,
        )

    @staticmethod
    def _check_field_specification_dict(file_options_dict):
        """Check the file_options_dict for valid inputs.

        Args:
            file_options_dict (dict): Dictionary containing the field description for the
                                      physical fields of interest that should be read-in.
        """
        required_keys_lst = ['target_time_lst', 'physical_field_dict', 'geometric_target']
        if not set(required_keys_lst).issubset(set(file_options_dict.keys())):
            raise KeyError(
                "The option 'file_options_dict' within the data_processor section must at least "
                f"contain the following keys: {required_keys_lst}. "
                f"You only provided: {file_options_dict.keys()}. Abort..."
            )

        required_field_keys = [
            'vtk_array_type',
            'vtk_field_label',
            'vtk_array_type',
            'field_components',
        ]

        if not set(required_field_keys).issubset(
            set(file_options_dict['physical_field_dict'].keys())
        ):
            raise KeyError(
                "The option 'physical_field_dict' within the data_processor section must at least "
                f"contain the following keys: {required_field_keys}. You only provided the keys: "
                f"{file_options_dict['physical_field_dict'].keys()}"
            )

    def _get_raw_data_from_file(self):
        """Read-in EnSight files using the vtkEnsightGoldBinaryReader."""
        # Set vtk reader object as raw file data
        self.raw_file_data = vtk.vtkEnSightGoldBinaryReader()
        self.raw_file_data.SetCaseFileName(self.file_path)
        self.raw_file_data.Update()

    def _filter_and_manipulate_raw_data(self):
        """Filter the ensight raw data for desired time steps."""
        # determine the correct time vector depending on input specification
        if self.target_time_lst[0] == "from_experimental_data":
            # get unique time list
            time_lst = list(set(self.experimental_data[self.time_label_experimental]))
        else:
            time_lst = self.target_time_lst

        # loop over different time-steps here.
        time_lst = sorted(time_lst)
        for time_value in time_lst:
            vtk_data_per_time_step = self._vtk_from_ensight(time_value)

            # check if data was found
            if not vtk_data_per_time_step:
                break

            # get ensight field from specified coordinates
            processed_data_per_time_step_interpolated = self._get_field_values_by_coordinates(
                vtk_data_per_time_step,
                time_value,
            )

            # potentially append new time step result to result array
            if self.processed_data.size > 0:
                self.processed_data = np.hstack(
                    (self.processed_data, processed_data_per_time_step_interpolated)
                )
            else:
                self.processed_data = processed_data_per_time_step_interpolated

    def _get_field_values_by_coordinates(
        self,
        vtk_data_obj,
        time_value,
    ):
        """Interpolate the vtk field to the coordinates from experimental data.

        Args:
            vtk_data_obj (obj): VTK ensight object that contains that solution fields of
                                     interest
            time_value (float): Current time value at which simulation should be evaluated

        Returns:
            interpolated_data (np.array): Array of field values at specified coordinates/geometric
                                          sets
        """
        if self.geometric_target[0] == "experimental_data":
            response_data = self._get_data_from_experimental_coordinates(
                vtk_data_obj,
                time_value,
            )
        elif self.geometric_target[0] == "geometric_set":
            response_data = self._get_data_from_geometric_set(
                vtk_data_obj,
            )
        else:
            raise ValueError(
                "Geometric target for ensight vtk must be either 'geometric_set' or"
                f"'experimental_data'. You provided '{self.geometric_target[0]}'. Abort..."
            )

        return response_data

    def _get_data_from_experimental_coordinates(self, vtk_data_obj, time_value):
        """Interpolate the ensight/vtk field to experimental coordinates.

        Args:
            vtk_data_obj (obj): VTK ensight object that contains that solution fields of
                                     interest
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
                raise ValueError("Please provide 3d coordinates in the observation data")
            experimental_coordinates_for_snapshot = np.concatenate(
                experimental_coordinates_for_snapshot, axis=1
            )
        # interpolate vtk solution to experimental coordinates
        interpolated_data = DataProcessorEnsight._interpolate_vtk(
            experimental_coordinates_for_snapshot,
            vtk_data_obj,
            self.vtk_array_type,
            self.vtk_field_label,
            self.vtk_field_components,
        )

        return interpolated_data

    def _get_data_from_geometric_set(
        self,
        vtk_data_obj,
    ):
        """Get entire fields for desired components.

        Args:
            vtk_data_obj (obj): VTK ensight object that contains that solution fields of
                                     interest
        Returns:
            data (np.array): Array of field values for nodes of geometric set
        """
        geometric_set = self.geometric_target[1]

        # get node coordinates by geometric set, loop over all topologies
        for nodes in self.geometric_set_data['node_topology']:
            if nodes['topology_name'] == geometric_set:
                nodes_of_interest = nodes['node_mesh']

        for lines in self.geometric_set_data['line_topology']:
            if lines['topology_name'] == geometric_set:
                nodes_of_interest = lines['node_mesh']

        for surfs in self.geometric_set_data['surface_topology']:
            if surfs['topology_name'] == geometric_set:
                nodes_of_interest = surfs['node_mesh']

        for vols in self.geometric_set_data['volume_topology']:
            if vols['topology_name'] == geometric_set:
                nodes_of_interest = vols['node_mesh']

        both = set(nodes_of_interest).intersection(
            self.geometric_set_data['node_coordinates']['node_mesh']
        )
        indices = [self.geometric_set_data['node_coordinates']['node_mesh'].index(x) for x in both]
        geometric_set_coordinates = [
            self.geometric_set_data['node_coordinates']['coordinates'][index] for index in indices
        ]

        # interpolate vtk solution to experimental coordinates
        interpolated_data = DataProcessorEnsight._interpolate_vtk(
            geometric_set_coordinates,
            vtk_data_obj,
            self.vtk_array_type,
            self.vtk_field_label,
            self.vtk_field_components,
        )

        return interpolated_data

    @staticmethod
    def _interpolate_vtk(
        coordinates, vtk_data_obj, vtk_array_type, vtk_field_label, vtk_field_components
    ):
        """Interpolate the vtk solution field to given coordinates.

        Args:
            coordinates (np.array): Coordinates at which the vtk solution field should be
                                    interpolated at
            vtk_data_obj (obj): VTK ensight object that contains that solution fields of
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
        probe_filter_obj.SetSourceData(vtk_data_obj)

        # Evaluate/interpolate this filter at experimental coordinates
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

    def _vtk_from_ensight(self, target_time):
        """Load a vtk-object from the ensight file.

        Args:
            target_time (float): Time the field should be evaluated on

        Returns:
            vtk_solution_field (obj)
        """
        # Ensight contains different "timesets" which are containers for the actual data
        number_of_timesets = self.raw_file_data.GetTimeSets().GetNumberOfItems()

        for num in range(number_of_timesets):
            time_set = vtk_to_numpy(self.raw_file_data.GetTimeSets().GetItem(num))
            # Find the timeset that has more than one entry as sometimes there is an empty dummy
            # timeset in the ensight file (seems to be an artifact)
            if time_set.size > 1:
                # if the keyword `last` was provided, get the last timestep
                if target_time == 'last':
                    self.raw_file_data.SetTimeValue(time_set[-1])
                else:
                    timestep = time_set.flat[np.abs(time_set - target_time).argmin()]

                    if np.abs(timestep - target_time) > self.time_tol:
                        raise RuntimeError(
                            "Time not within tolerance"
                            f"Target time: {target_time}, selected time: {timestep},"
                            f"tolerance {self.time_tol}"
                        )
                    self.raw_file_data.SetTimeValue(timestep)

                self.raw_file_data.Update()
                input_vtk = self.raw_file_data.GetOutput()

                # the data is in the first block of the vtk object
                vtk_solution_field = input_vtk.GetBlock(0)

        return vtk_solution_field

    @staticmethod
    def read_geometry_coordinates(external_geometry_obj):
        """Read geometry of interest.

        This method uses the QUEENS external geometry module.

        Args:
            external_geometry_obj (pqueens.baci_dat_geometry)

        Returns:
            dict: set with BACI topology
        """
        # read in the external geometry
        external_geometry_obj.main_run()

        geometric_set_data = {
            "node_topology": external_geometry_obj.node_topology,
            "line_topology": external_geometry_obj.line_topology,
            "surface_topology": external_geometry_obj.surface_topology,
            "volume_topology": external_geometry_obj.volume_topology,
            "node_coordinates": external_geometry_obj.node_coordinates,
            "element_centers": external_geometry_obj.element_centers,
            "element_topology": external_geometry_obj.element_topology,
        }

        return geometric_set_data
