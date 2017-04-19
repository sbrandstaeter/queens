import numpy as np
from pqueens.randomfields.random_field_gen_fourier import RandomFieldGenFourier

class RandomFieldGenFourier1D(RandomFieldGenFourier):
    """ Fourier series based 1-d random field generator.

    Generator for 1-dimensional random fields based on a Fourier series
    expansion.

    Attributes:

        marginal_distribution (scipy.stats.rv_continuous): marginal distribution
            of random field
        corr_length (float): correlation length of random field
        energy_frac (float): energy fraction retianed by Fourier series expansion
        field_bbox (np.array): bouding box for field
        num_ex_term_per_dim (int): number of expansion temrm per dimension
        num_terms (int): number of terms in expansion


    """

    def __init__(self,marginal_distribution,corr_length,energy_frac,field_bbox,
                 num_ex_term_per_dim,num_terms):
        """
        Args:

            marginal_distribution (scipy.stats.rv_continuous): marginal
                distribution of random field
            corr_length (float): correlation length of random field
            energy_frac (float): energy fraction retianed by Fourier series
                expansion
            field_bbox (np.array): bouding box for field
            num_ex_term_per_dim (int): number of expansion term per dimension
            num_terms (int): number of terms in expansion
        """

        # call superclass constructor first
        super().__init__(marginal_distribution,corr_length,energy_frac,
                         field_bbox,1,num_ex_term_per_dim,num_terms)

        # setup truncation of fourier expansion
        num_ck=0
        sum_ck=0
        index=0

        self.kb = np.empty([self.m,1])
        print(self.m)
        for k1 in range(0,self.m):
            sum_ck=sum_ck+(self.compute_expansion_coefficient( \
                           k1,self.largest_length,self.corr_length))
            num_ck=num_ck+1
            self.kb[index,0]=k1
            index=index+1

        if sum_ck < energy_frac:
            raise RuntimeError('Error: not converged try again')

        self.act_energy_frac=sum_ck

        # find out stochastic dimension based on kb
        self.stoch_dim = len(self.kb)*2


    def gen_sample_gauss_field(self,loc, phase_angles):
        """ Generate sample of Gaussian field.

        Compute realization of standard Gaussian field based on passed phase
        angles phase_angles and return values of the realization at loc.

        Args:
            loc (np.array): location at which field is evaluated
            phase_angles (np.array): pseudo random phase angles for field
                generation
        Returns:
            np.array: vector containing values of realization at specified
                locations
        """
        if(len(phase_angles) is not self.stoch_dim):
            raise RuntimeError('Number of random phase angles does not match '
                               'stochastic dimension of the field!')

        #if len(loc[0,:]) is not 1:
        #    raise RuntimeError('Location vector must have one dimensions!')

        # reorder phase angles in matrix
        xi = np.reshape(phase_angles, (-1,2))
        tempgp=0
        for i in range(self.kb.shape[0]):
            wk1=self.kb[i,0]*np.pi/self.largest_length
            tempgp=tempgp+np.sqrt((super().compute_expansion_coefficient( \
            self.kb[i,0],self.largest_length,self.corr_length)))* \
            (xi[i,0]*np.cos(wk1*loc[:])+xi[i,1]*np.sin(wk1*loc[:]))
        return tempgp
