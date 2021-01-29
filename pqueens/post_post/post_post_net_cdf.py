import glob
import os
import numpy as np
import xarray as xr
from pqueens.post_post.post_post import PostPost
from pqueens.database.mongodb import MongoDB


class PostPostNetCDF(PostPost):
    """
    Class for post-post-processing net-CDF data (xarrays)

    Attributes:
        experiment_name (str): Name of the QUEENS simulation
        db (obj): Database object
        coordinate_labels (list): Names/labels of coordinates/dimensions on which the
                                  solution/output of the simulation is defined on
        replace_non_numerics (): Variable that defines whether and how we should replace NaNs and
                                 Infs in the solution fields

    Returns:
        PostPostNetCDF_obj (obj): Instance of the PostPostNetCDF class

    """

    def __init__(
        self,
        delete_data_flag,
        file_prefix,
        experiment_name,
        db,
        coordinate_labels,
        output_label,
        replace_non_numerics,
    ):
        super(PostPostNetCDF, self).__init__(delete_data_flag, file_prefix)
        self.experiment_name = experiment_name
        self.db = db
        self.coordinate_labels = coordinate_labels
        self.output_label = output_label
        self.replace_non_numerics = replace_non_numerics

    @classmethod
    def from_config_create_post_post(cls, options):
        """ Create post_post routine from problem description

        Args:
            options (dict): input options

        Returns:
            post_post (obj): PostPostNetCDF object
        """
        post_post_options = options['options']
        delete_data_flag = post_post_options['delete_field_data']
        file_prefix = post_post_options['file_prefix']
        experiment_name = options['config']['global_settings']['experiment_name']
        coordinate_labels = options['config']['method']['method_options']['coordinate_labels']
        output_label = options['config']['method']['method_options']['output_label']
        db = MongoDB.from_config_create_database(options['config'])
        replace_non_numerics = options['config']['driver']['driver_params']['post_post'].get(
            'replace_non_numeric_values'
        )

        return cls(
            delete_data_flag,
            file_prefix,
            experiment_name,
            db,
            coordinate_labels,
            output_label,
            replace_non_numerics,
        )

    def read_post_files(self):
        """
        Read-in netCDF files and interpolate fields at experimental data locations.
        Return then only the interpolated field values on these location in form of a
        np.array.

        """
        # Read in all netCDF data experiment directory as a xarray
        prefix_expr = '*' + self.file_prefix + '*.nc'
        files_of_interest = os.path.join(self.output_dir, prefix_expr)
        post_file = glob.glob(files_of_interest)
        # glob returns arbitrary list -> need to sort the list before using
        post_file.sort()

        if len(post_file) > 1:
            raise IOError(
                "Several netCDF-output files per simulation run are not supported at the moment"
            )

        try:
            # here we save the entire netCDF file to a xarray variable
            post_out = xr.open_dataset(post_file[0])

            # very simple error check
            if not post_out:
                self.error = True
                self.result = None

        except IOError:
            print('Could not read-in simulation data...')
            self.error = True
            self.result = None

        self.error = False
        # interpolate results by coordinates such that they correspond to the experimental data
        # coordinates
        self.result = self._get_field_values_by_coordinates(post_out)

    def _get_field_values_by_coordinates(self, post_out):
        """
        Interpolate xarray solution field of simulation at the coordinates that correspond to the
        experimental data such that we only return an array of these interpolated values instead of
        the entire array.

        Args:
            post_out (xarray): Solution field of the simulation run in form of a xarray with
                               coordinates and solution variable defined on the coordinate
                               dimensions

        Returns:
            result_vec (np.array): Vector with interpolated solution field values that correspond to
                                   the experimental coordinates

        """
        # load experimental data dict form database (we saved this under batch 1)
        experimental_data_dict = self.db.load(self.experiment_name, '1', 'experimental_data')
        # get corresponding simulation data
        coordinates_dict = {key: experimental_data_dict[key] for key in self.coordinate_labels}

        if len(coordinates_dict.keys()) == 1:
            x = xr.DataArray(coordinates_dict[self.coordinate_labels[0]], dims=self.output_label)
            coordinates_dict = {self.coordinate_labels[0]: x}
        elif len(coordinates_dict.keys()) == 2:
            x = xr.DataArray(coordinates_dict[self.coordinate_labels[0]], dims=self.output_label)
            y = xr.DataArray(coordinates_dict[self.coordinate_labels[1]], dims=self.output_label)
            coordinates_dict = {self.coordinate_labels[0]: x, self.coordinate_labels[1]: y}
        else:
            raise NotImplementedError(
                'Currently we do not support coordinate dimensions greater than two! Abort...'
            )

        result_vec = post_out.interp(
            coordinates_dict
        ).__xarray_dataarray_variable__.values.flatten()

        result_vec = self._replace_non_numeric_values(result_vec)

        return result_vec

    def _replace_non_numeric_values(self, result_vec):
        """
        Method to handle/replace non-numeric values in the solution fields or its interpolation.
        If the variable 'replace_non_numerics' is not specified, the default settings does not
        replace any values.

        Args:
            result_vec (np.array): Solution array that potentially contains non-numerics values

        Returns:
            result_vec (np.array): Solution array after taking care of non-numeric values

        """
        if self.replace_non_numerics is not None:
            result_vec[np.isnan(result_vec)] = self.replace_non_numerics
        else:
            raise NotImplementedError(
                f'The replacement method {self.replace_non_numerics} is not '
                f'implemented! Please choose an implemented '
                f'replacement-method for '
                f'non-numeric data! Abort ...'
            )
        return result_vec
