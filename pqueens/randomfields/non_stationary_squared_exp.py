""" This should be a docstring """

import numpy as np


class NonStationarySquaredExp():
    """ This should be a docstring """
    def __init__(self, non_stat_opt):
        self.corr_length = non_stat_opt['corr_length']
        self.rel_std = non_stat_opt['rel_std']
        self.mean_fun = non_stat_opt['mean_fun']
        self.mean_fun_params = non_stat_opt['mean_fun_params']
        self.num_points = non_stat_opt['num_points']
        self.num_realizations = non_stat_opt['num_samples']
        self.mean = None
        self.K_mat = None
        self.cholesky = None
        self.realizations = None
        self.x_vec = None
        self.nugget = 1E-07

    def main_run(self):
        """ This should be a docstring """
        self.calculate_mean_fun()
        self.calculate_covariance_matrix_and_cholseky()
        self.calculate_realizations()

# ----------------------------- AUXILIARY METHODS -----------------------------
    def calculate_mean_fun(self):
        """ This should be a docstring """
        self.x_vec = np.linspace(0, 1, self.num_points, endpoint=True)
        if self.mean_fun == 'inflow_parabola':
            # Parabola that has its maximum at x = 0
            self.mean = 4 * self.mean_fun_params[0] * (-(self.x_vec-0.5)**2 + 0.25)
        else:
            raise RuntimeError('Only inflow parabola implemented at the moment!')

    def calculate_covariance_matrix_and_cholseky(self):
        """ This should be a docstring """
        K_mat = np.zeros((self.num_points, self.num_points))
        for num1, x_one in enumerate(self.x_vec):
            for num2, x_two in enumerate(self.x_vec):
                K_mat[num1, num2] = np.exp(-(x_one - x_two)**2/(2*self.corr_length**2))
        self.K_mat = K_mat + self.nugget * np.eye(self.num_points)
        self.cholesky = np.linalg.cholesky(self.K_mat)

    def calculate_realizations(self):
        """ This should be a docstring """
        self.realizations = np.zeros((self.num_realizations, self.num_points))
        for num in range(self.num_realizations):
            np.random.seed(num)  # fix a specific random seed to make runs repeatable
            rand = np.random.normal(0, 1, self.num_points)
            self.realizations[num, :] = self.mean * (1 + self.rel_std * np.dot(self.cholesky, rand))
            self.realizations[num, 0] = 0  # BCs
            self.realizations[num, -1] = 0  # BCs
