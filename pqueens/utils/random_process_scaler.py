import numpy as np
import abc


class Scaler(metaclass=abc.ABCMeta):
    """
    Base class for general scaling classes. The purpose of these classes
    is the scaling of training data for, e.g., machine learning approaches
    or other subsequent analysis.

    Attributes:
        mean (np.array): Mean-values of the data-matrix (column-wise)
        standard_deviation (np.array): Standard deviation of the data-matrix (per column)
 
    Returns:
        Instance of the Scaler Class (obj)

    """

    def __init__(self, mean, standard_deviation):
        self.mean = mean
        self.standard_deviation = standard_deviation

    @classmethod
    def from_config_create_scaler(cls, scaler_settings):
        """
        Create scaler from problem description

        Args:
            scaler_name (str):
            scaler_settings (str):

        Returns:
            Instance of Scaler class (obj)

        """
        scaler_dict = {'standard_scaler': StandardScaler, 'identity_scaler': IdentityScaler}

        scaler_class = scaler_dict[scaler_settings["type"]]

        # initiate some attributes
        mean = None
        standard_deviation = None

        return scaler_class.from_config_create_scaler(scaler_settings, mean, standard_deviation)

    @abc.abstractmethod
    def fit(self):
        """
        Fit/calculate the scaling based on the input samples
        """
        pass

    @abc.abstractmethod
    def transform(self):
        """
        Conduct the scaling transformation on the input samples.
        """
        pass

    @abc.abstractmethod
    def inverse_transform_mean(self):
        """
        Conduct the inverse transformation for the mean / the mean function
        of the random process trained on the scaled training data.
        """
        pass

    @abc.abstractmethod
    def inverse_transform_std(self):
        """
        Conduct the inverse transformation for the posterior standard deviation of the
        random process trained on the scaled training data.
        """
        pass


class StandardScaler(Scaler):
    """
    Scaler for standardization of data. In case a stochastic process in trained on the scaled data,
    inverse rescaling is implemented to recover the correct mean and standard deviation prediction
    for the posterior process.

    Returns:
        instance of the StandardScaler (obj)

    """

    def __init__(self, mean, standard_deviation):
        super(StandardScaler, self).__init__(mean, standard_deviation)

    @classmethod
    def from_config_create_scaler(cls, scaler_settings, mean, standard_deviation):
        """
        Create a Standard scaler object based on the problem description

        Args:
            settings (dict):  Settings of the scaler
            mean (np.array): Array containing mean values for data matrix columns
            standard_deviation (np.array): Array containing standard deviations for
                                           data matrix columns

        Returns:
            StandardScaler instance (obj)

        """
        return cls(mean, standard_deviation)

    def fit(self, x_mat):
        """
        Fit/calculate the scaling based on the input samples

        Args:
            x_mat (np.array): Data matrix that should be standardized

        Returns:
            None

        """
        self.mean = np.mean(x_mat)
        self.standard_deviation = np.std(x_mat)

    def transform(self, x_mat):
        """
        Conduct the scaling transformation on the data matrix.

        Args:
            x_mat (np.array): Data matrix that should be standardized

        Returns:
            transformed_data (np.array): Transformed data-array

        """
        transformed_data = (x_mat - self.mean) / self.standard_deviation
        return transformed_data

    def inverse_transform_mean(self, x_mat):
        """
        Conduct the inverse scaling transformation on the data matrix.

        Args:
            x_mat (np.array): Data matrix that should be standardized

        Returns:
            transformed_data (np.array): Transformed data-array

        """
        transformed_data = x_mat * self.standard_deviation + self.mean

        return transformed_data

    def inverse_transform_std(self, x_mat):
        """
        Conduct the inverse scaling transformation on the standard deviation data of the
        random process.

        Args:
            x_mat (np.array): Data matrix that should be standardized

        Returns:
            transformed_data (np.array): Transformed data-array

        """
        transformed_data = x_mat * self.standard_deviation

        return transformed_data


class IdentityScaler(Scaler):
    """
    The identity scaler is shares the interfaces of other scalers but does nothing to the data.
    """

    def __init__(self, mean, standard_deviation):
        super(IdentityScaler, self).__init__(mean, standard_deviation)

    @classmethod
    def from_config_create_scaler(cls, scaler_settings, mean, standard_deviation):
        """
        Create a Standard scaler object based on the problem description

        Args:
            scaler_settings (dict): Settings of the scaler
            mean (np.array): Array containing mean values for data matrix columns
            standard_deviation (np.array): Array containing standard deviations for
                                           data matrix columns

        Returns:
            IdentityScaler instance (obj)

        """

        return cls(mean, standard_deviation)

    def fit(x_mat):
        """
        Fit/calculate the scaling based on the input samples

        Args:
            x_mat (np.array): Data matrix that should be standardized

        Returns:
            None

        """
        pass

    def transform(self, x_mat):
        """
        Conduct the scaling transformation on the data matrix.

        Args:
            x_mat (np.array): Data matrix that should be standardized

        Returns:
            transformed_data (np.array): Transformed data-array

        """

        return x_mat

    def inverse_transform_mean(self, x_mat):
        """
        Conduct the inverse scaling transformation on the data matrix.

        Args:
            x_mat (np.array): Data matrix that should be standardized

        Returns:
            transformed_data (np.array): Transformed data-array

        """
        return x_mat

    def inverse_transform_std(self, x_mat):
        """
        Conduct the inverse scaling transformation on the standard deviation data of the
        random process.

        Args:
            x_mat (np.array): Data matrix that should be standardized

        Returns:
            transformed_data (np.array): Transformed data-array

        """
        return x_mat
