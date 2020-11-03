import re
from pqueens.external_geometry.external_geometry import ExternalGeometry


class BaciDatExternalGeometry(ExternalGeometry):
    """
    Class to read in external geometries based on BACI-dat files.

    Args:
        path_to_dat_file (str): Path to dat file from which the geometry should be extracted
        list_geometric_sets (list): List of geometric sets that should be extracted
        node_topology (dict): Topology of edges/nodes (here: mesh nodes not FEM nodes)
        line_topology (dict): Topology of line sets
        surface_topology (dict): Topology of surfaces
        volume_topology (dict): Topology of volumes
        node_coordinates (dict): Coordinates of mesh nodes

    Attributes:
        path_to_dat_file (str): Path to dat file from which the geometry should be extracted
        list_geometric_sets (list): List of geometric sets that should be extracted
        current_dat_section (str): String that encodes the current section in the dat file as
                                   file is read line-wise
        design_description (dict): First section of the dat-file as dictionary to summarize the
                                   geometry components
        node_topology (dict): Topology of edges/nodes (here: mesh nodes not FEM nodes)
        line_topology (dict): Topology of line components
        surface_topology (dict): Topology of surfaces
        volume_topology (dict): Topology of volumes
        desired_dat_sections (dict): Dictionary that holds only desired dat-sections and
                                     geometric sets within these sections so that we can skip
                                     undesired parts of the dat-file
        nodes_of_interest (list): List that contains all (mesh) nodes that are part of a desired
                                  geometric component
        node_coordinates (dict): Dictionary that holds the desired nodes as well as their
                                 corresponding geometric coordinates

    Returns:
        geometry_obj (obj): Instance of BaciDatExternalGeometry class

    """

    dat_sections = [
        'DESIGN DESCRIPTION',
        'DNODE-NODE TOPOLOGY',
        'DLINE-NODE TOPOLOGY',
        'DSURF-NODE TOPOLOGY',
        'DVOL-NODE TOPOLOGY',
        'NODE COORDS',
    ]
    section_match_dict = {
        "DPOINT": "DNODE-NODE TOPOLOGY",
        "DLINE": "DLINE-NODE TOPOLOGY",
        "DSURFACE": "DSURF-NODE TOPOLOGY",
        "DVOL": "DVOL-NODE TOPOLOGY",
    }

    def __init__(
        self,
        path_to_dat_file,
        list_geometric_sets,
        node_topology,
        line_topology,
        surface_topology,
        volume_topology,
        node_coordinates,
    ):

        super(BaciDatExternalGeometry, self).__init__()
        self.path_to_dat_file = path_to_dat_file
        self.list_geometric_sets = list_geometric_sets
        self.current_dat_section = None
        self.design_description = {}
        self.node_topology = node_topology
        self.line_topology = line_topology
        self.surface_topology = surface_topology
        self.volume_topology = volume_topology
        self.desired_dat_sections = {}
        self.nodes_of_interest = None
        self.node_coordinates = node_coordinates

    @classmethod
    def from_config_create_external_geometry(cls, config):
        """
        Create BaciDatExternalGeometry object from problem description

        Args:
            config (dict): Problem description

        Returns:
            geometric_obj (obj): Instance of BaciDatExternalGeometry

        """
        interface_name = config['model'].get('interface')
        driver_name = config[interface_name].get('driver')
        path_to_dat_file = config[driver_name]['driver_params']['input_template']
        list_geometric_sets = config['external_geometry'].get('list_geometric_sets')
        node_topology = {"node_mesh": [], "node_topology": []}
        line_topology = {"node_mesh": [], "line_topology": []}
        surface_topology = {"node_mesh": [], "surface_topology": []}
        volume_topology = {"node_mesh": [], "volume_topology": []}
        node_coordinates = {"node_mesh": [], "coordinates": []}

        return cls(
            path_to_dat_file,
            list_geometric_sets,
            node_topology,
            line_topology,
            surface_topology,
            volume_topology,
            node_coordinates,
        )

    # --------------- child methods that must be implemented --------------------------------------
    def read_external_data(self):
        self._read_geometry_from_dat_file()

    def organize_sections(self):
        self._get_desired_dat_sections()

    def finish_and_clean(self):
        pass

    # -------------- helper methods ---------------------------------------------------------------
    def _read_geometry_from_dat_file(self):
        """
        Read the dat-file line by line to be memory efficient.
        Only save desired information.

        Returns:
            None

        """
        with open(self.path_to_dat_file) as my_dat:
            # read dat file line-wise
            for line in my_dat:
                line = line.strip()

                match_bool = self._get_current_dat_section(line)
                # skip comments outside of section definition
                if line[0:2] == '//' or match_bool:
                    pass
                else:
                    self._get_design_description(line)
                    self._get_only_desired_topology(line)
                    self._get_only_desired_coordinates(line)

    def _get_current_dat_section(self, line):
        """
        Check if the current line starts a new section in the dat-file.
        Update self.current_dat_section if new section was found. If the current line is the
        actual section identifier, return a True boolean.

        Args:
            line (str): Current line of the dat-file

        Returns:
            bool (boolean): True or False depending if current line is the section match

        """
        # regex for the sections
        section_name_re = re.compile('^-+([^-].+)$')
        match = section_name_re.match(line)
        # get the current section of the dat file
        if match:
            # remove whitespaces and horizontal line
            section_string = line.strip('-')
            section_string = section_string.strip()
            # check for comments
            if line[:2] == '//':
                return True
            # ignore comment pattern after actual string
            elif section_string.strip('//') in self.dat_sections:
                self.current_dat_section = section_string
                return True
            else:
                self.current_dat_section = None
                return True
        else:
            return False

    def _check_if_in_desired_dat_section(self):
        """
        Return True if we are in a dat-section that contains the desired geometric set.

        Returns:
            Boolean

        """
        if self.current_dat_section in self.desired_dat_sections.keys():
            return True
        else:
            return False

    def _get_desired_dat_sections(self):
        """
        Get the dat-sections (and its identifier) that contain the desired geometric sets.

        Returns:
            None

        """
        # initialize keys with empty lists
        for geo_set in self.list_geometric_sets:
            self.desired_dat_sections[self.section_match_dict[geo_set.split()[0]]] = []

        # write desired geometric set to corresponding section key
        for geo_set in self.list_geometric_sets:
            self.desired_dat_sections[self.section_match_dict[geo_set.split()[0]]].append(geo_set)

    def _get_topology(self, line):
        """
        Get the geometric topology by extracting and grouping (mesh) nodes that
        belong to the desired geometric sets and save them to their topology class.

        Args:
            line (str): Current line of the dat-file

        Returns:
            None

        """
        # get edges
        if self.current_dat_section == 'DNODE-NODE TOPOLOGY':
            if ' '.join(line.split()[2:4]) in self.desired_dat_sections['DNODE-NODE TOPOLOGY']:
                node_list = line.split()

                self.node_topology['node_mesh'].append(int(node_list[1]))
                self.node_topology['node_topology'].append(int(node_list[3]))

        # get lines
        elif self.current_dat_section == 'DLINE-NODE TOPOLOGY':
            if ' '.join(line.split()[2:4]) in self.desired_dat_sections['DLINE-NODE TOPOLOGY']:
                node_list = line.split()
                self.line_topology['node_mesh'].append(int(node_list[1]))
                self.line_topology['line_topology'].append(int(node_list[3]))

        # get surfaces
        elif self.current_dat_section == 'DSURF-NODE TOPOLOGY':
            if ' '.join(line.split()[2:4]) in self.desired_dat_sections['DSURF-NODE TOPOLOGY']:
                node_list = line.split()
                self.surface_topology['node_mesh'].append(int(node_list[1]))
                self.surface_topology['surface_topology'].append(int(node_list[3]))

        # get volumes
        elif self.current_dat_section == 'DVOL-NODE TOPOLOGY':
            if ' '.join(line.split()[2:4]) in self.desired_dat_sections['DVOL-NODE TOPOLOGY']:
                node_list = line.split()
                self.volume_topology['node_mesh'].append(int(node_list[1]))
                self.volume_topology['volume_topology'].append(int(node_list[3]))

    def _get_design_description(self, line):
        """
        Extract a short geometric description from the dat-file

        Args:
            line (str): Current line of the dat-file

        Returns:
            None

        """
        # get the overall design description of the problem at hand
        if self.current_dat_section == 'DESIGN DESCRIPTION':
            design_list = line.split()
            if len(design_list) != 2:
                raise IndexError(
                    f'Unexpected number of list entries in design '
                    f'description! The '
                    'returned list should have length 2 but the returned '
                    'list was {design_list}.'
                    'Abort...'
                )

            self.design_description[design_list[0]] = design_list[1]

    def _get_only_desired_topology(self, line):
        """
        Check if the current dat-file sections contains desired geometric sets. Skip the topology
        extraction if the current section does not contain a desired geometric set, anyways.

        Args:
            line (str): Current line of the dat-file

        Returns:
            None

        """
        # skip lines that are not part of a desired section
        desired_section_boolean = self._check_if_in_desired_dat_section()

        if not desired_section_boolean:
            pass
        else:
            # get topology groups
            self._get_topology(line)

    def _get_only_desired_coordinates(self, line):
        """
        Get coordinates of nodes that belong to a desired geometric set.

        Args:
            line (str): Current line of the dat-file

        Returns:
            None

        """
        if self.current_dat_section == 'NODE COORDS':
            node_list = line.split()
            if self.nodes_of_interest is None:
                self._get_nodes_of_interest()

            if self.nodes_of_interest is not None:
                if int(node_list[1]) in self.nodes_of_interest:
                    self._get_coordinates_of_desired_geometric_sets(node_list)

    def _get_coordinates_of_desired_geometric_sets(self, node_list):
        """
        Extract node and coordinate information of the current dat-file line.

        Args:
            node_list (list): Current line of the dat-file

        Returns:
            None

        """
        self.node_coordinates['node_mesh'].append(int(node_list[1]))
        nodes_as_float_list = [float(value) for value in node_list[3:6]]
        self.node_coordinates['coordinates'].append(nodes_as_float_list)

    def _get_nodes_of_interest(self):
        """
        Based on the extracted topology, get a unique list of all (mesh) nodes that are part of
        the extracted topology.

        Returns:
            None

        """
        node_mesh_nodes = self.node_topology['node_mesh']
        line_mesh_nodes = self.line_topology['node_mesh']
        surf_mesh_nodes = self.surface_topology['node_mesh']
        vol_mesh_nodes = self.volume_topology['node_mesh']

        nodes_of_interest = node_mesh_nodes + line_mesh_nodes + surf_mesh_nodes + vol_mesh_nodes

        # make node_list unique
        self.nodes_of_interest = list(set(nodes_of_interest))
