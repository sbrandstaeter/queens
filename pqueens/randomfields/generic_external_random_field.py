"""TODO_doc."""

import numpy as np
import scipy as sp


class GenericExternalRandomField:
    """Generic random field class for random fields.

    Generic random field class for random fields on externally defined geometries.

    Attributes:
        corr_length (float): Hyperparameter for the correlation length (a.t.m. only one).
        std_hyperparam_rf (float): Hyperparameter for standard-deviation of random field.
        mean_fun_params (lst): List of parameters for mean function parameterization of
                               random field.
        num_samples (int): Number of samples/realizations of the random field.
        num_points (int): Number of discretization points of the random field.
        mean (np.array): Vector that contains the discretized mean function of the
                                   random field.
        K_mat (np.array): Covariance matrix of the random field.
        cholesky_decomp_covar_mat (np.array): Cholesky decomposition of covariance matrix.
        realizations (np.array): Realization/sample of the random field.
        fixed_one_dim_coords_vector (np.array): Fixed coordinate vector for discretization of the
                                                random field (depreciated).
        nugget_variance_rf (float): Nugget variance for the random field (lower bound for
                                    diagonal values of the covariance matrix).
        mean_fun_type (str): Type of mean function of the random field.
        external_geometry_obj (obj): External geometry object.
        external_definition (dict): External definition of the random field.
        random_field_coordinates (np.array): Matrix with row-wise coordinate values of the
                                             random field discretization.
        eigen_vecs_mat (np.array): Eigenvector matrix of covariance.
        eigen_vals_vec (np.array): Vector of eigenvalues of covariance matrix.
        weighted_eigen_val_mat_truncated (np.array): Truncated and with eigenvalues weighted
                                                     eigen-representation of the covariance
                                                     matrix.
        dimension: TODO_doc

    Returns:
        Instance of GenericExternalRandomField class
    """

    def __init__(
        self,
        corr_length=None,
        std_hyperparam_rf=None,
        mean_fun_params=None,
        num_samples=None,
        external_definition=None,
        external_geometry_obj=None,
        mean_fun_type=None,
        dimension=None,
    ):
        """TODO_doc.

        Args:
            corr_length: TODO_doc
            std_hyperparam_rf: TODO_doc
            mean_fun_params: TODO_doc
            num_samples: TODO_doc
            external_definition: TODO_doc
            external_geometry_obj: TODO_doc
            mean_fun_type: TODO_doc
            dimension: TODO_doc
        """
        self.corr_length = corr_length
        self.std_hyperparam_rf = std_hyperparam_rf
        self.mean_fun_params = mean_fun_params
        self.num_samples = num_samples
        self.num_points = None  # not num_samples!
        self.mean = None
        self.K_mat = None
        self.cholesky_decomp_covar_mat = None
        self.realizations = None
        self.fixed_one_dim_coords_vector = None
        self.nugget_variance_rf = 1e-9
        self.mean_fun_type = mean_fun_type
        self.external_geometry_obj = external_geometry_obj
        self.external_definition = external_definition
        self.random_field_coordinates = None
        self.eigen_vecs_mat = None
        self.eigen_vals_vec = None
        self.weighted_eigen_val_mat_truncated = None
        self.dimension = dimension

    def main_run(self):
        """Main run of the external random field class.

        Main run of the external random field class that finds a lower
        dim representation for the random field and returns the basis
        and lower dim (truncated) coefficient vector of the field
        representation.
        """
        self.calculate_mean_fun()
        self.calculate_covariance_matrix_and_cholesky()
        self.calculate_random_coef_matrix()

    # ----------------------------- AUXILIARY METHODS -----------------------------
    def calculate_mean_fun(self):
        """Calculate the mean function of the random field.

        Calculate the mean function of the random field and store the
        discretized representation.
        """
        if self.mean_fun_type == 'inflow_parabola':
            self.fixed_one_dim_coords_vector = np.linspace(0, 1, self.num_points, endpoint=True)
            # Parabola that has its maximum at x = 0
            self.mean = (
                4
                * self.mean_fun_params[0]
                * (-((self.fixed_one_dim_coords_vector - 0.5) ** 2) + 0.25)
            )
        elif self.mean_fun_type == 'constant':  # TODO quick option for testing should be extra
            # get name of geometric set the current rf is defined on
            geometric_set_name = self.external_definition['external_instance']
            field_type = self.external_definition['type']

            if field_type == 'material':
                # get element centers and coordinate
                # TODO atm we assume only one random field this should be generalized
                coordinates_random_field = self.external_geometry_obj.element_centers

            else:
                # loop over all topologies of interest
                topology_total = (
                    self.external_geometry_obj.node_topology
                    + self.external_geometry_obj.line_topology
                    + self.external_geometry_obj.surface_topology
                    + self.external_geometry_obj.volume_topology
                )

                # filter topology for geometric nodes for the current random field
                random_field_nodes = sorted(
                    (
                        [
                            topology['node_mesh']
                            for topology in topology_total
                            if topology['topology_name'] == geometric_set_name
                        ]
                    )[0]
                )

                # get coordinates of nodes for topology for current random field
                coordinates_random_field = []
                index = 0
                for node in random_field_nodes:
                    while self.external_geometry_obj.node_coordinates['node_mesh'][index] != node:
                        index += 1
                    coordinates_random_field.append(
                        self.external_geometry_obj.node_coordinates['coordinates'][index]
                    )
                    index += 1

            self.random_field_coordinates = np.array(coordinates_random_field)[:, : self.dimension]
            self.num_points = self.random_field_coordinates.shape[0]
            self.mean = self.mean_fun_params[0] * np.ones(self.random_field_coordinates.shape[0])
        else:
            raise RuntimeError('Only inflow parabola and constant implemented at the moment!')

    def calculate_covariance_matrix_and_cholesky(self):
        """Build covariance matrix and calculate a Cholesky decomposition.

        Based on the kernel description of the random field, build its
        covariance matrix using the external geometry and coordinates.
        Afterwards, calculate the Cholesky decomposition.
        """
        K_mat = np.zeros((self.num_points, self.num_points))
        # here we assume a specific kernel, namely a rbf kernel
        for num1, x_one in enumerate(self.random_field_coordinates):
            for num2, x_two in enumerate(self.random_field_coordinates):
                K_mat[num1, num2] = self.std_hyperparam_rf**2 * np.exp(
                    -(np.linalg.norm(x_one - x_two) ** 2) / (2 * self.corr_length**2)
                )

        self.K_mat = K_mat + self.nugget_variance_rf * np.eye(self.num_points)
        self.cholesky_decomp_covar_mat = np.linalg.cholesky(self.K_mat)

        # decompose and truncate the random field
        self._decompose_and_truncate_random_field()

    def _decompose_and_truncate_random_field(self):
        """Decompose and truncate the random field.

        Decompose and then truncate the random field according to
        desired variance fraction that should be covered/explained by
        the truncation.
        """
        # compute eigendecomposition
        # TODO we should use the information about the Cholesky decomp
        eig_val, eig_vec = sp.linalg.eigh(self.K_mat)
        self.eigen_vals_vec = np.real(eig_val)
        self.eigen_vecs_mat = np.real(eig_vec)

        sum_val = 0
        sum_eigenval = np.sum(self.eigen_vals_vec)
        # calculate m, which is the truncated length and covers 98% of variance
        for num, eigenval in reversed(list(enumerate(self.eigen_vals_vec))):
            sum_val += eigenval
            variance_fraction = sum_val / sum_eigenval
            num_eigen = num
            if variance_fraction > 0.95:  # TODO pull this out to json file
                break
        # truncated eigenfunction base
        eigen_vec_mat_red = self.eigen_vecs_mat[:, num_eigen:]

        # truncated eigenvalues
        eig_val_vec_red = self.eigen_vals_vec[num_eigen:]

        # truncated diagonal eigenvalue matrix
        eigen_val_red_diag_mat = np.diagflat(eig_val_vec_red)

        # weight the eigenbasis with the eigenvalues
        self.weighted_eigen_val_mat_truncated = np.dot(
            eigen_vec_mat_red, np.sqrt(eigen_val_red_diag_mat)
        )

    def calculate_random_coef_matrix(self):
        """Provide the random coefficients of the truncated field.

        The actual field is not build here but will be reconstructed
        from the coefficient matrix and the truncated basis.
        """
        # TODO this should be changed to new truncated version (code already goes in this method)
        # TODO copy here content of `univariate_field_generator_factory` staticmethod
        self.realizations = np.zeros((self.num_samples, self.num_points))
        if self.mean_fun_type == 'inflow_parabola':  # TODO quick option for testing -> should
            for num in range(self.num_samples):
                # be extra class
                # TODO this part is still in the old representation
                np.random.seed(num)  # fix a specific random seed to make runs repeatable
                rand = np.random.normal(0, 1, self.num_points)
                self.realizations[num, :] = self.mean * (
                    1 + self.std_hyperparam_rf * np.dot(self.cholesky_decomp_covar_mat, rand)
                )
                self.realizations[num, 0] = 0  # BCs
                self.realizations[num, -1] = 0  # BCs

        elif self.mean_fun_type == 'constant':
            np.random.seed(1)  # TODO: pull this out to json
            dim_truncation = self.weighted_eigen_val_mat_truncated.shape[1]

            # will be written to the db externally
            self.realizations = np.random.normal(0, 1, (dim_truncation, self.num_samples))
        else:
            raise RuntimeError('Only inflow parabola and constant implemented at the moment!')
