"""TODO_doc."""

import numpy as np

from pqueens.randomfields.generic_external_random_field import GenericExternalRandomField
from pqueens.randomfields.random_field_gen_fourier_1d import RandomFieldGenFourier1D
from pqueens.randomfields.random_field_gen_fourier_2d import RandomFieldGenFourier2D
from pqueens.randomfields.random_field_gen_fourier_3d import RandomFieldGenFourier3D
from pqueens.randomfields.random_field_gen_KLE_1d import RandomFieldGenKLE1D
from pqueens.randomfields.random_field_gen_KLE_2d import RandomFieldGenKLE2D
from pqueens.randomfields.random_field_gen_KLE_3d import RandomFieldGenKLE3D


class UniVarRandomFieldGeneratorFactory(object):
    """TODO_doc: add one-line explanation.

    Class that is currently used for the generation of random fields and
    which gets called in other modules such as the *monte_carlo_iterator*. This
    class is basically a wrapper for different existing random field
    definitions.

    Returns:
        rf (obj): Instance of a random field class
    """

    # TODO: we should clean this up and update the rfs to the QUEENS coding style and architecture

    def create_new_random_field_generator(
        marg_pdf=None,
        corrstruct=None,
        spatial_dimension=None,
        corr_length=None,
        energy_frac=None,
        field_bbox=None,
        num_terms_per_dim=None,
        total_terms=None,
        std_hyperparam_rf=None,
        mean_fun_params=None,
        mean_fun_type=None,
        num_samples=None,
        external_geometry_obj=None,
        external_definition=None,
        dimension=None,
    ):
        """Create random field generator based on arguments.

        Args:
            marg_pdf (obj): Marginal probability distribution of the random field (for spectral
                            definition only)
            corrstruct (str): Correlation structure or type of random field that should be
                              initialized
            spatial_dimension (int): Spatial dimension of the random field
            corr_length (float): Hyperparameter for the correlation length (a.t.m. only one)
            energy_frac (float): Energy fraction of random field truncation for spectral
                                 representations
            field_bbox (np.array): Box in which the random field should be realized (for spectral
                              representation only and without external definition)
            num_terms_per_dim (int): Number of terms per dimension (for spectral decomposition only)
            total_terms (int): Total Number of terms (spectral decomposition only)
            std_hyperparam_rf (float): Hyperparameter for standard-deviation of random field
            mean_fun_params (lst): List of parameters for mean function parameterization
                                   of random field
            mean_fun_type (str): Type of mean function that should be used
            num_samples (int): Number of *samples*/*field_realizations* of the random field
            external_geometry_obj (obj): Instance of the external geometry class
            external_definition (dict): External definition of the random fields
            dimension: TODO_doc

        Returns:
            rf (obj): Instance of a random field generator class
        """
        if corrstruct == 'generic_external_random_field':
            rf = GenericExternalRandomField(
                corr_length=corr_length,
                std_hyperparam_rf=std_hyperparam_rf,
                mean_fun_params=mean_fun_params,
                num_samples=num_samples,
                external_definition=external_definition,
                external_geometry_obj=external_geometry_obj,
                mean_fun_type=mean_fun_type,
                dimension=dimension,
            )

        elif corrstruct == 'squared_exp':
            if spatial_dimension == 1:
                rf = RandomFieldGenFourier1D(
                    marg_pdf,
                    corr_length,
                    energy_frac,
                    field_bbox,
                    num_terms_per_dim,
                    total_terms,
                )
            elif spatial_dimension == 2:
                rf = RandomFieldGenFourier2D(
                    marg_pdf,
                    corr_length,
                    energy_frac,
                    field_bbox,
                    num_terms_per_dim,
                    total_terms,
                )
            elif spatial_dimension == 3:
                rf = RandomFieldGenFourier3D(
                    marg_pdf,
                    corr_length,
                    energy_frac,
                    field_bbox,
                    num_terms_per_dim,
                    total_terms,
                )
            else:
                raise ValueError(
                    f"Spatial dimension must be either 1, 2, or 3, not {spatial_dimension}"
                )

        elif corrstruct == 'exp':
            if spatial_dimension == 1:
                rf = RandomFieldGenKLE1D(
                    marg_pdf,
                    corr_length,
                    energy_frac,
                    field_bbox,
                    spatial_dimension,
                    num_terms_per_dim,
                    total_terms,
                )
            elif spatial_dimension == 2:
                rf = RandomFieldGenKLE2D(
                    marg_pdf,
                    corr_length,
                    energy_frac,
                    field_bbox,
                    spatial_dimension,
                    num_terms_per_dim,
                    total_terms,
                )
            elif spatial_dimension == 3:
                rf = RandomFieldGenKLE3D(
                    marg_pdf,
                    corr_length,
                    energy_frac,
                    field_bbox,
                    spatial_dimension,
                    num_terms_per_dim,
                    total_terms,
                )
            else:
                raise ValueError(
                    f'Spatial dimension must be either 1,2, or 3, not {spatial_dimension}'
                )
        else:
            raise RuntimeError(
                'Auto-correlation structure has to be'
                f' either "squared_exp" or "exp", not {corrstruct}'
            )
        return rf

    factory = staticmethod(create_new_random_field_generator)

    @staticmethod
    def calculate_one_truncated_realization_of_all_fields(
        database, job_id, experiment_name, batch, experiment_dir, random_fields_lst, driver_name
    ):
        """TODO_doc: add one-line explanation.

        This method gets called in the driver and calculates one realization
        of all involved random fields from the in the db stored truncated
        basis, the random coefficients matrix and the current job number.
        (Driver realizes one input sample, such that the *job_id* is used to
        identify the current sample from the sample matrix.)

        Args:
            database (obj): Database instance
            job_id (int): Job ID number
            experiment_name (str): Name of the current QUEENS experiment
            batch (int): Batch number
            experiment_dir (str): Path to QUEENS experiment directory
            random_fields_lst (lst): List of random field definitions
            driver_name (str): Name of the driver for current analysis

        Returns:
            realized_random_fields_lst (lst): List containing
            *field_realizations* of involved random fields
        """
        # load random field representation
        truncated_random_field_representation_dict = database.load(
            experiment_name, '1', 'truncated_random_fields'
        )

        # load current job
        job = database.load(
            experiment_name,
            batch,
            'jobs_' + driver_name,
            {'id': job_id, 'experiment_dir': experiment_dir, 'experiment_name': experiment_name},
        )

        # calculate high dim realization of random field from basis functions
        realized_random_fields_lst = []
        for random_field in random_fields_lst:
            random_field_name = random_field[0]
            random_field_definition = random_field[1]
            # random field realization: mean_fun + realization with mean zero (dot product)
            random_vec = truncated_random_field_representation_dict[random_field_name][1] + np.dot(
                truncated_random_field_representation_dict[random_field_name][0],
                job['params'][random_field_name],
            )

            updated_random_field_dict = {
                "name": random_field_name,
                "external_definition": random_field_definition,
                "values": random_vec,
            }
            realized_random_fields_lst.append(updated_random_field_dict)

        return realized_random_fields_lst
