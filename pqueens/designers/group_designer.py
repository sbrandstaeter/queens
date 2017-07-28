import numpy as np
import random
import math
from pqueens.designers.abstract_designer import AbstractDesigner

class GroupDesigner(AbstractDesigner):
    """ GroupDesigner generate 1 tensor GP of dimension (number of combinaision x
    number of dimensions x number of samples), whom each row is a different
    combinaison of X and X_tilde, necessary to create groups to compute all
    generelized Sensitivity Indices. This designer can also be a start for doing
    group sampling, see [1].

    References:
    [1] A. Saltelli, M. Ratto, T. Andres, F. Campolongo, T. Cariboni, D. Galtelli,
        M. Saisana, S. Tarantola. "GLOBAL SENSITIVITY ANALYSIS. The Primer",
        109 - 121,
        ISBN : 978-0-470-05997-5

    Attributes:
        self.num_samples (int): number of design points
        self.GP (np.array): array with all samples/design points

    Example:
    GD = GroupDesigner(params,num_samples)
    Y = GD.get_all_samples()

    """
    def __init__(self,params,num_samples):
        """
        Args:
            params (dict):
                Two samples X and X_tilde defining the input space of dimension d, from
                which we can make groups of variables.
            num_samples (int) : Number of desired (random) samples

        """
        numparams = len(params)
        self.dim = len(params['X'])
        self.nb_combi = (self.dim+2+math.factorial(self.dim)//(2*math.factorial(self.dim-2)))
        self.num_samples = num_samples
        self.gp = []
        self.gp_tilde = []
        self.X_Stock = np.ones((2,num_samples,self.dim), dtype = np.float64)
        self.X = np.ones((num_samples,self.dim), dtype = np.float64)
        self.X_tilde = np.ones((num_samples,self.dim), dtype = np.float64)

        n = 0
        for key, value in params.items():
            i = 0
            if n < numparams:
                for k, v  in value.items():
                    self.X_Stock[n,:,i] = v
                    i = i + 1
                n = n + 1
            self.X = self.X_Stock[0,:,:].transpose()
            self.X_tilde = self.X_Stock[1,:,:].transpose()

    def generate_liste_combinaison(self,seq,k):
        """
        Function which generates all different possible combinaisons of length k
        from the coordinates of a given vector.

        Attributes:
            seq (np.array) : vector from which we want to draw the combinaisons
            k (int) : length of the combinaisons we want, k has to <= len(seq)
        """
        p=[]
        imax = 2**len(seq)-1
        for i in range(imax+1):
            liste_combi = []
            jmax = len(seq)-1
            for j in range(jmax+1):
                if (i>>j)&1==1:
                    liste_combi.append(seq[j])
            if len(liste_combi) == k:
                p.append(liste_combi)
        return p

    def get_all_samples(self):
        """ Generate all possible combinaisons from X and X_tilde
        necessary to compute the Sensitivity Indices with Algorithm 1 in [1] """
        seq1 = range(0,self.dim)
        p = []
        for i in range(1,self.dim+1):
            p.append(self.generate_liste_combinaison(seq1,i))
        p.pop()
        GP = np.ones((self.num_samples,2**self.dim,self.dim), dtype = float)
        GP[:,0,:] = self.X.transpose()
        GP[:,2**self.dim-1,:] = self.X_tilde.transpose()
        GP_temp = np.ones((self.num_samples,self.dim), dtype = float)

        k = 1
        for i in range(self.dim-1):
            nb_combi = math.factorial(self.dim)//(math.factorial(i+1)*math.factorial(self.dim-i-1))
            for j in range(nb_combi):
                GP_temp[:,p[i][j]] = self.X[p[i][j]].transpose()
                GP_temp[:,p[self.dim-1-i-1][nb_combi-j-1]] = self.X_tilde[p[self.dim-1-i-1][nb_combi-j-1]].transpose()
                GP[:,k,:] = GP_temp
                k = k +1

        return GP
