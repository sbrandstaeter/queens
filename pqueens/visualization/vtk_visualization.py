"""Visualization class for vtk.

A module that provides utilities and a class for visualization with vtk.
"""
from pathlib import Path

from pqueens.utils.path_utils import check_if_path_exists
from pqueens.utils.vtk_utils import (
    add_point_array_to_vtu_object,
    export_vtu_file,
    get_node_coordinates_from_vtu_file,
    load_vtu_file,
)


class VTKVisualization:
    """Visualization class for exporting fields using vtk.

    Attributes:
       vtk_output_path (str): Path to directory where to save the vtk outputs.
       geometry_file (str): Path to geometry file
       save_every (int): Declares, how often the visualization will be done (default: 1)

    Returns:
        VTKVisualization (obj) : Instance of the VTKVisualization Class
    """

    def __init__(self, output_dir, geometry_file, save_every, field_name, node_coordinates):
        """Initialize VTKVisualization object.

        Args:
            output_dir (Path): Output directory of QUEENS
            geometry_file (str): Path to vtu template
            save_every (int): Which iteration to save the field
            field_name (str): name of the field being exported
            node_coordinates (np.array): Nodal coordinates of the vtu geometry file
        """
        self.output_dir = output_dir
        self.geometry_file = geometry_file
        self.save_every = save_every
        self.field_name = field_name
        self.node_coordinates = node_coordinates

    @classmethod
    def from_config_create(cls, config, field_name):
        """Create the visualization object from the config.

        Args:
            config (dict): Dictionary containing the problem description
            field_name (str): Name of the field to be exported

        Returns:
            Instance of VTKVisualization (obj)
        """
        visualization_options = config["visualization"].get("visualization_options")

        output_dir = Path(visualization_options.get("output_dir"))

        geometry_file = visualization_options.get("geometry_file", None)

        if geometry_file is None:
            raise KeyError("Keyword 'geometry' missing in the input-file!")
        check_if_path_exists(geometry_file)

        save_every = visualization_options.get("save_every", 1)
        node_coordinates = get_node_coordinates_from_vtu_file(geometry_file)

        return cls(
            output_dir=output_dir,
            geometry_file=geometry_file,
            save_every=save_every,
            field_name=field_name,
            node_coordinates=node_coordinates,
        )

    def write_vtk_file(self, field, iteration):
        """Export data to vtk format and save with geometry as a file.

        Args:
            field (np.array): field data for its given discretization
            iteration (int): current iteration of the iterator
        """
        if iteration % self.save_every == 0:
            vtk_obj = load_vtu_file(self.geometry_file)
            add_point_array_to_vtu_object(vtk_obj, self.field_name, field)
            file_name = self.output_dir.joinpath(f"{self.field_name}_{iteration}.vtu")
            export_vtu_file(vtk_obj, str(file_name))
