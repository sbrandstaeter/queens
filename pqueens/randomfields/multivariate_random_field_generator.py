import numpy as np
import scipy
from  pqueens.randomfields.univariate_field_generator_factory import UniVarRandomFieldGeneratorFactory

class MultiVariateRandomFieldGenerator(object):
    """ Generator of samples of multivariate cross-correlated random fields

    Class for the generation of samples from Gaussian and
    and non-Gaussian multi variate random fields using translation process
    theory. The generation of the cross-correlated random fields is based on
    the approach described in

    Vorechovsky, Miroslav. "Simulation of simply cross correlated random fields
    by series expansion methods." Structural safety 30.4 (2008): 337-363.

    Attributes:
        var_dim (int):  how many variates do we have
        prob_dists (list): vector of marginal probability distribution of the fields
        stoch_dim       stochastic dimension, i.e., number of random variable needed to generate samples
        my_univ_rfs (list):  univaraite random field generators
        pre_factor  factor to multiply independent random phase angles with
                   (product of eigenvector and values of cross-correlation matrix)

    """

    # crosscorr (np.array)
    #  marginal_distributions (list)
    def __init__(self,marginal_distributions, num_fields, crosscorr, spatial_dimension,
                 corr_struct, corr_length,energy_frac,field_bbox,
                 num_terms_per_dim,total_terms):

        # do some sanity checks
        if (num_fields == len(crosscorr[:,0]) and
            num_fields == len(crosscorr[0,:])):
            self.var_dim = num_fields
        else:
            raise RuntimeError('Size of cross-correlation matrix must '
                               'match number of fields')

        if(num_fields != len(marginal_distributions)):
            raise RuntimeError('Number of input distributions does not '
                               ' match number of univariate fields')

        temp=0
        self.my_univ_rfs =[]
        for i in range (self.var_dim):
            self.my_univ_rfs.append(UniVarRandomFieldGeneratorFactory.\
                                   create_new_random_field_generator(
                                   marginal_distributions[i],
                                   spatial_dimension,
                                   corr_struct,
                                   corr_length,
                                   energy_frac,
                                   field_bbox,
                                   num_terms_per_dim,
                                   total_terms))

            temp = temp+self.my_univ_rfs[i].stoch_dim

        self.stoch_dim=temp

        # calculate pre_factor_ matrix to obtain block correlated random
        # phase angles/amplitude according to Vorechovsky 2008

        # check wehter cross-correlation matrix is semi-positive definite
        #[~,r] = chol(crosscorr)
        #if r:
        #    error('Error: The provided crosscorr matrix is not positive definite')
        # do this only to check whether cross correlation matrix is
        # semi-positive definite
        np.linalg.cholesky(crosscorr)

        # eigenvalue decomposition of cross-correlation matrix
        # TODO check if eigenvectors are correct
        lambda_c, phi_c = np.linalg.eig(crosscorr)
        #print('crosscorr{}'.format(crosscorr))
        print('lambda_c{}'.format(lambda_c))
        #print('phi_c{}'.format(phi_c))
        #temp=[]

        # use dimension of first field for now
        temp=np.diag(np.ones((self.my_univ_rfs[0].stoch_dim,)))
        #print('self.my_univ_rfs[0].stoch_dim{}'.format(self.my_univ_rfs[0].stoch_dim))
        #print('temp{}'.format(temp))


        #phi_d=sparse(np.kron(phi_c,temp))
        phi_d=np.kron(phi_c,temp)
        #print('phi_d{}'.format(phi_d))
        #print('phi_d shape{}'.format(phi_d.shape))


        # block diagonal matrix lambda_d
        #lamda_c2=np.diag(lambda_c) # get diagonal entries
        #print('lamda_c2{}'.format(lamda_c2))
        #lamdba_c=cell(self.var_dim,1)
        lamdba_c3 =[]
        for i in range(len(lambda_c)):
            lamdba_c3.append(lambda_c[i]*temp)

        print('lamdba_c3{}'.format(lamdba_c3))

        #lambda_d=sparse(blkdiag(lamdba_c3[:]))
        lambda_d=scipy.linalg.block_diag(*lamdba_c3)
        #numpy.set_printoptions(threshold=numpy.nan)
        #print('lamdba_d{}'.format(lambda_d))
        #print('lamdba_d shape{}'.format(lambda_d.shape))

        #self.pre_factor=sparse(phi_d*(lambda_d**(1/2)))
        self.pre_factor=(phi_d*(lambda_d**(1/2)))
        #print('self.pre_factor{}'.format(self.pre_factor))
        #print('self.pre_factor shape{}'.format(self.pre_factor.shape))


    def get_stoch_dim(self):
         # GetStochDim return stochastic dimension of the field
         return (self.stoch_dim)


    def evaluate_field_at_location(self,x, xi):
        # EvaluateFieldAtLocation generate realization of random field
        # and compute value at input location
        new_vals=np.zeros((x.shape[0],self.var_dim))
        helper = np.dot(self.pre_factor,xi.T)
        
        my_xi=helper.reshape(-1,self.var_dim)

        for i in range (self.var_dim):
            new_vals[:,i]=self.my_univ_rfs[i].evaluate_field_at_location(x, my_xi[:,i])
        return new_vals
