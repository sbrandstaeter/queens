from pqueens.randomfields.non_stationary_squared_exp import NonStationarySquaredExp

from pqueens.randomfields.random_field_gen_fourier_1d import RandomFieldGenFourier1D
from pqueens.randomfields.random_field_gen_fourier_2d import RandomFieldGenFourier2D
from pqueens.randomfields.random_field_gen_fourier_3d import RandomFieldGenFourier3D

from pqueens.randomfields.random_field_gen_KLE_1d import RandomFieldGenKLE1D
from pqueens.randomfields.random_field_gen_KLE_2d import RandomFieldGenKLE2D
from pqueens.randomfields.random_field_gen_KLE_3d import RandomFieldGenKLE3D


class UniVarRandomFieldGeneratorFactory(object):

    def create_new_random_field_generator(my_marg_pdf, non_stat_opt):
        """ Create random field generator based on arguments """
        # unpack the dictionary
        # TODO intermediate solution: all fields should read in a dict rather than individual attributes
        corr_struct = non_stat_opt['corrstruct']
        if corr_struct == 'non_stationary_squared_exp':
            rf = NonStationarySquaredExp(non_stat_opt)
        else:
            spatial_dimension = non_stat_opt['dimension']
            corr_struct = non_stat_opt['corrstruct']
            corr_length = non_stat_opt['corr_length']
            energy_frac = non_stat_opt['energy_frac']
            field_bbox = non_stat_opt['field_bbox']
            num_terms_per_dim = non_stat_opt['num_terms_per_dim']
            total_terms = non_stat_opt['total_terms']

        if corr_struct == 'squared_exp':
            if spatial_dimension == 1:
                rf = RandomFieldGenFourier1D(my_marg_pdf, corr_length,
                                             energy_frac,
                                             field_bbox,
                                             num_terms_per_dim,
                                             total_terms)
            elif spatial_dimension == 2:
                rf = RandomFieldGenFourier2D(my_marg_pdf, corr_length,
                                             energy_frac,
                                             field_bbox,
                                             num_terms_per_dim,
                                             total_terms)
            elif spatial_dimension == 3:
                rf = RandomFieldGenFourier3D(my_marg_pdf, corr_length,
                                             energy_frac,
                                             field_bbox,
                                             num_terms_per_dim,
                                             total_terms)
            else:
                raise ValueError('Spatial dimension must be either 1,2, or 3,'
                                 ' not {}'.format(spatial_dimension))

        elif corr_struct == 'exp':
            if spatial_dimension == 1:
                rf = RandomFieldGenKLE1D(my_marg_pdf, corr_length, energy_frac,
                                         field_bbox, spatial_dimension,
                                         num_terms_per_dim, total_terms)
            elif spatial_dimension == 2:
                rf = RandomFieldGenKLE2D(my_marg_pdf, corr_length, energy_frac,
                                         field_bbox, spatial_dimension,
                                         num_terms_per_dim, total_terms)
            elif spatial_dimension == 3:
                rf = RandomFieldGenKLE3D(my_marg_pdf, corr_length, energy_frac,
                                         field_bbox, spatial_dimension,
                                         num_terms_per_dim, total_terms)
            else:
                raise ValueError('Spatial dimension must be either 1,2, or 3,'
                                 ' not {}'.format(spatial_dimension))
        elif corr_struct == 'non_stationary_squared_exp':
            pass
        else:
            raise RuntimeError('Autocorrelation structure has to be either'
                               ' "squared_exp" or "exp", not {}'.format(corr_struct))
        return rf


    factory = staticmethod(create_new_random_field_generator)
