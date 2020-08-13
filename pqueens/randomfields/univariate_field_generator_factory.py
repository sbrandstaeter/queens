from pqueens.randomfields.non_stationary_squared_exp import NonStationarySquaredExp

from pqueens.randomfields.random_field_gen_fourier_1d import RandomFieldGenFourier1D
from pqueens.randomfields.random_field_gen_fourier_2d import RandomFieldGenFourier2D
from pqueens.randomfields.random_field_gen_fourier_3d import RandomFieldGenFourier3D

from pqueens.randomfields.random_field_gen_KLE_1d import RandomFieldGenKLE1D
from pqueens.randomfields.random_field_gen_KLE_2d import RandomFieldGenKLE2D
from pqueens.randomfields.random_field_gen_KLE_3d import RandomFieldGenKLE3D


class UniVarRandomFieldGeneratorFactory(object):
    def create_new_random_field_generator(
        marg_pdf=None,
        corrstruct=None,
        spatial_dimension=None,
        corr_length=None,
        energy_frac=None,
        field_bbox=None,
        num_terms_per_dim=None,
        total_terms=None,
        rel_std=None,
        mean_fun_params=None,
        num_points=None,
        num_realizations=None,
    ):
        """ Create random field generator based on arguments """
        if corrstruct == 'non_stationary_squared_exp':
            rf = NonStationarySquaredExp(
                corr_length=corr_length,
                rel_std=rel_std,
                mean_fun_params=mean_fun_params,
                num_points=num_points,
                num_realizations=num_realizations,
            )

        elif corrstruct == 'squared_exp':
            if spatial_dimension == 1:
                rf = RandomFieldGenFourier1D(
                    marg_pdf, corr_length, energy_frac, field_bbox, num_terms_per_dim, total_terms,
                )
            elif spatial_dimension == 2:
                rf = RandomFieldGenFourier2D(
                    marg_pdf, corr_length, energy_frac, field_bbox, num_terms_per_dim, total_terms,
                )
            elif spatial_dimension == 3:
                rf = RandomFieldGenFourier3D(
                    marg_pdf, corr_length, energy_frac, field_bbox, num_terms_per_dim, total_terms,
                )
            else:
                raise ValueError(
                    'Spatial dimension must be either 1,2, or 3,'
                    ' not {}'.format(spatial_dimension)
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
                    'Spatial dimension must be either 1,2, or 3,'
                    ' not {}'.format(spatial_dimension)
                )
        else:
            raise RuntimeError(
                'Autocorrelation structure has to be either'
                ' "squared_exp" or "exp", not {}'.format(corrstruct)
            )
        return rf

    factory = staticmethod(create_new_random_field_generator)
