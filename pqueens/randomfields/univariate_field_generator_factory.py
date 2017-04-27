
from pqueens.randomfields.random_field_gen_fourier_1d import RandomFieldGenFourier1D
from pqueens.randomfields.random_field_gen_fourier_2d import RandomFieldGenFourier2D
from pqueens.randomfields.random_field_gen_fourier_3d import RandomFieldGenFourier3D

from pqueens.randomfields.random_field_gen_KLE_1d import RandomFieldGenKLE1D
from pqueens.randomfields.random_field_gen_KLE_2d import RandomFieldGenKLE2D
from pqueens.randomfields.random_field_gen_KLE_3d import RandomFieldGenKLE3D


class UniVarRandomFieldGeneratorFactory(object):
    # Create based on arguments:
    def create_new_random_field_generator(my_marg_pdf, spatial_dimension,
                                          corr_struct, corr_length,
                                          energy_frac, field_bbox,
                                          num_terms_per_dim, total_terms):

        if corr_struct == 'squared_exp':
            if spatial_dimension == 1:
                rf = RandomFieldGenFourier1D(my_marg_pdf,corr_length,
                                             energy_frac,
                                             field_bbox,
                                             num_terms_per_dim,
                                             total_terms)
            elif spatial_dimension ==2:
                rf = RandomFieldGenFourier2D(my_marg_pdf,corr_length,
                                             energy_frac,
                                             field_bbox,
                                             num_terms_per_dim,
                                             total_terms)
            elif spatial_dimension == 3:
                rf = RandomFieldGenFourier3D(my_marg_pdf,corr_length,
                                             energy_frac,
                                             field_bbox,
                                             num_terms_per_dim,
                                             total_terms)
            else:
                raise ValueError('Spatial dimension must be either 1,2, or 3, not {}'.format(spatial_dimension))

        elif corr_struct == 'exp':
            if spatial_dimension == 1:
                rf = RandomFieldGenKLE1D(my_marg_pdf,corr_length,energy_frac,
                                         field_bbox,spatial_dimension,
                                         num_terms_per_dim,total_terms)
            elif spatial_dimension == 2:
                #raise NotImplementedError()
                rf = RandomFieldGenKLE2D(my_marg_pdf,corr_length,energy_frac,
                                         field_bbox,spatial_dimension,
                                         num_terms_per_dim,total_terms)
            elif spatial_dimension == 3:
                rf = RandomFieldGenKLE3D(my_marg_pdf,corr_length,energy_frac,
                                         field_bbox,spatial_dimension,
                                         num_terms_per_dim,total_terms);
            else:
                raise ValueError('Spatial dimension must be either 1,2, or 3, not {}'.format(spatial_dimension))
        else:
            raise RuntimeError('Autocorrelation structure has to be either "squared_exp" or "exp", not {}'.format(corr_struct))
        return rf


    factory = staticmethod(create_new_random_field_generator)
