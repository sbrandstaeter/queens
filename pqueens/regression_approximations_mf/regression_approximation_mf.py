import abc


class RegressionApproximationMF(metaclass=abc.ABCMeta):
    """Base class for multi-fidelity regression approximations.

    Regression approxiamtion are regression models/approaches that are
    called regression approximations within QUEENS to avoid the term
    model.
    """

    @classmethod
    def from_options(cls, approx_options, Xtrain, Ytrain):
        """Create multi-fideltiy approximation from options dict.

        Args:
            approx_options (dict): Dictionary with approximation options
            Xtrain (list):         List with training input arrays
            Ytrain (list):         List with training output arrays

        Returns:
            RegressionApproximationMF: Multi-Fidelity regression approximation object
        """
        from .mf_icm_gp_regression import MF_ICM_GP_Regression
        from .mf_nar_gp_regression_2_levels import MF_NAR_GP_Regression_2_Levels
        from .mf_nar_gp_regression_3_levels import MF_NAR_GP_Regression_3_Levels

        approx_dict = {
            'mf_icm_gp_approximation_gpy': MF_ICM_GP_Regression,
            'mf_nar_gp_approximation_gpy_2_levels': MF_NAR_GP_Regression_2_Levels,
            'mf_nar_gp_approximation_gpy_3_levels': MF_NAR_GP_Regression_3_Levels,
        }

        approximation_class = approx_dict[approx_options["type"]]

        return approximation_class.from_options(approx_options, Xtrain, Ytrain)

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def predict(self, Xnew):
        pass

    @abc.abstractmethod
    def predict_f(self, Xnew):
        pass

    @abc.abstractmethod
    def predict_f_samples(self, Xnew, num_samples):
        pass
