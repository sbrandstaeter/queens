import scipy
import numpy as np
from pqueens.randomfields.univariate_random_field_generator import UnivariateRandomFieldSimulator

class RandomFieldGenKLE(UnivariateRandomFieldSimulator):
    """ Karhuenen Loeve  based random field generator


     Random field generation based on Karhunen-Loeve expansion using the
     analytic solution of the Fredholm equation presented in [#f1]_.

    .. rubric:: Footnotes
    .. [#f1] Zhang, D., & Lu, Z. (2004). An efficient, high-order perturbation
             approach for flow in random porous media via Karhunen-Loeve and
             polynomial expansions. Journal of Computational Physics, 194(2),
             773-794. http://doi.org/10.1016/j.jcp.2003.09.015

    Attributes:
        m (int): number of terms in expansion in each direction
        trunc_thres (int): truncation threshold for Fourier series
        largest_length (double): length of random field (for now equal in all
                                 dimensions based on largest dimension of
                                 bounding box
        corr_length (double): correlation length of field
                              (so far only isotropic fields)
        w_n (np.array): roots of characteristic functions
        lambda_n (np.array): eigenvalues of Fredholm equation
        eps (double): = 0.000000001 tolerance for root finding
                      TODO make static or set relative to corr_length
    """
    EPS = 0.000000001

    def __init__(self,marginal_distribution,corr_length,energy_frac,field_bbox,
                 dimension,num_ex_term_per_dim,num_terms):

        self.m = None
        self.trunc_thres = None
        self.largest_length = None
        self.corr_length = None
        self.w_n = None
        self.lambda_n = None
        # call superclass  first
        super().__init__(marginal_distribution)

        # sanity checks are done in factory
        self.spatial_dim=dimension

        san_check_bbox=field_bbox.shape
        if san_check_bbox[0] is not self.spatial_dim*2:
            raise ValueError('field bounding box must be size {} and not {}'
                              .format(self.spatial_dim*2,san_check_bbox[0]))

        self.bounding_box=field_bbox

        # compute largest length and size of random field for now.
        # reshape bounding box so that each dimension is in new row
        bbox = np.reshape(self.bounding_box, (self.spatial_dim,2))

        # compute the maximum
        self.largest_length = bbox.max(0).max(0)

        if energy_frac<0 or energy_frac>1 :
            raise ValueError('energy fraction must be between 0 and 1.')

        self.des_energy_frac=energy_frac

        if(corr_length<=0):
            raise ValueError('Correlation length must be positive')

        self.corr_length=corr_length

        # based on the number of terms per dimension, we can have only
        # num_ex_term_per_dim^dim terms in total
        # TODO check if this is true
        if (num_ex_term_per_dim**self.spatial_dim > num_terms):
            raise ValueError('Number of terms in KLE expansion is too large. '
                             'Decrease number of terms or increase number of '
                             'terms per dimension')


        self.m=num_ex_term_per_dim
        self.trunc_thres=num_terms
        self.stoch_dim=num_terms



    def compute_roots_of_characteristic_equation(self, curr_dim):
        # ComputeRootsOfCharEq compute roots of characteristic equation (see (11) in [1])
        # argument curr_dim not really needed right now because we assume that the field is isotropic.
        # However, in the future this might change ...

        #init w_n
        w_n =np.zeros((self.m,1))
        index=0

        #compute w_n we need to compute the positive roots of equation (11) from [1]

        # we need to find the positive roots w_n of the following characteristic
        # function (which is implemented in compute_w )
        # 2*corr_length*w/(corr_length^2 w^2 -1) - tan(w*largest_length )

        # this function has poles which must be taken into account
        # while searching for roots using fzero(), where we can search
        # for roots between the poles. The pole of the first part (2*corr_length_*w/(corr_length_^2 w^2 -1))
        # is at sqrt(1/corr_length_) and the pole of the second part is
        # at n/2 *pi, where k=1,3,5,7, ...

        # check for root before the first pole of the tan function only
        # if the root of the first part lies before the root of the
        # second part.
        if (1/self.corr_length <np.pi/(self.largest_length*2)):
             w_n[index] = scipy.optimize.brentq(self.compute_w,
                                                (1/self.corr_length) +
                                                 self.EPS,np.pi /
                                                 (2*self.largest_length) -
                                                self.EPS)
             index = index+1


        # we need odd n only but want a total of m roots hence
        # subtract 1 from m because we have one addtional root through
        # additional pole from first part of characteristic equation
        for n in range(1, ((self.m-1)*2), 2):
            # check if root of first part lies in currrent considered interval
            if(n*np.pi/(self.largest_length*2) < (1/self.corr_length) and
              (n+2)*np.pi/(2*self.largest_length) > (1/self.corr_length) ):
                w_n[index] = scipy.optimize.brentq(self.compute_w,n*np.pi /
                                                   (2*self.largest_length) +
                                                    self.EPS,(1/self.corr_length) -
                                                    self.EPS)
                index=index+1
                w_n[index] = scipy.optimize.brentq(self.compute_w,
                                                   (1/self.corr_length) +
                                                   self.EPS,(n+2) * np.pi /
                                                   (2*self.largest_length)
                                                   -self.EPS)
                index=index+1
            else:
                w_n[index]= scipy.optimize.brentq(self.compute_w,
                                                  n*np.pi / (2*self.largest_length) +
                                                  self.EPS,
                                                  (n+2) * np.pi /
                                                  (2*self.largest_length) -
                                                  self.EPS)
                index=index+1

        return w_n


    def  compute_w(self,w):
        # compute_w characteristic function, see (12) in [1]
        return 2*self.corr_length*w/(self.corr_length**2*w**2-1) - \
               np.tan(w*self.largest_length)

    def  compute_eigen_function_vec(self,loc,dim):
        # compute_eigen_function_vec  compute eigenfunctions of Fredholm equation
        # fval = ComputeEigenfunctionVec(obj,loc,dim) returns the value of the eigenfunctions at the locations loc
        # dim denotes the current dimension
        x=loc[:,dim].reshape(-1,1)
        helper=(np.ones((len(x),1)))

        temp1 =1./(np.sqrt((self.corr_length**2 *
                   (np.kron((self.w_n[:,dim]),helper))**2+1) *
                   self.largest_length/2+self.corr_length))
        temp2 = self.corr_length*np.kron((self.w_n[:,dim]),helper) * \
               np.cos(np.kron((self.w_n[:,dim]),x)) \
               + np.sin(np.kron((self.w_n[:,dim]),x))
        return temp1*temp2
