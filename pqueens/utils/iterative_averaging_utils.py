import abc

import numpy as np


class IterativeAveraging(metaclass=abc.ABCMeta):
    """Base class for iterative averaging schemes.

    Attributes:
        current_average (np.array): Current average value
        new_value (np.array): New value for the averaging process
        rel_L1_change (float): Relative change in L1 norm of the average value
        rel_L2_change (float): Relative change in L2 norm of the average value
    """

    def __init__(self, current_average, new_value, rel_L1_change, rel_L2_change):
        self.current_average = current_average
        self.new_value = new_value
        self.rel_L1_change = rel_L1_change
        self.rel_L2_change = rel_L2_change

    @classmethod
    def from_config_create_iterative_averaging(cls, config, section_name=None):
        """
            Build a iterative averaging scheme from config
        Args:
            config (dict): Configuration dict
            section_name (str): Name of section where the averaging object is configured

        Returns:
            iterative averaging object

        """
        valid_options = ["moving_average", "polyak_averaging", "exponential_averaging"]
        if section_name:
            averaging_type = config[section_name].get("averaging_type")
        else:
            averaging_type = config.get("averaging_type")

        if averaging_type == "moving_average":
            return MovingAveraging.from_config_create_iterative_averaging(config, section_name)
        elif averaging_type == "polyak_averaging":
            return PolyakAveraging.from_config_create_iterative_averaging(config, section_name)
        elif averaging_type == "exponential_averaging":
            return ExponentialAveraging.from_config_create_iterative_averaging(config, section_name)
        else:
            raise NotImplementedError(
                f"Iterative averaging option '{averaging_type}' unknown. Valid options are"
                f"{valid_options}"
            )

    def _compute_rel_change(self, old_average, new_average):
        """Compute the relative changes (L1 and L2) between new and old
        average.

        Args:
            old_average (np.array): Old average value
            new_average (np.array): New average value
        """
        self.rel_L2_change = relative_change(old_average, new_average, L2_norm)
        self.rel_L1_change = relative_change(old_average, new_average, L1_norm)

    def update_average(self, new_value):
        """Compute the actual average. (Is scheme specific)

        Args:
            new_value (np.array): New observation for the averaging
        """
        if isinstance(new_value, (float, int)):
            new_value = np.array(new_value)
        if self.current_average is not None:
            old_average = self.current_average.copy()
            self.current_average = self.average_computation(new_value)
            self._compute_rel_change(old_average, self.current_average)
        else:
            # If it is the first observation
            self.current_average = new_value.copy()
        return self.current_average.copy()

    @abc.abstractclassmethod
    def average_computation(self):
        """Here the averaging approach is implemented."""
        pass


class MovingAveraging(IterativeAveraging):
    """
    Compute the moving average:
        :math:`x^{(j)}_{avg}=\\frac{1}{k}\\sum_{i=0}^{k-1}x^{(j-i)}`
    where :math: `k-1` is the number of values from previous iterations that are used

    Attributes:
        current_average (np.array): Current average value
        new_value (np.array): New value for the averaging process
        rel_L1_change (float): Relative change in L1 norm of the average value
        rel_L2_change (float): Relative change in L2 norm of the average value
        num_iter_for_avg (int): Number of samples in the averaging window
        data (list): List of the stored values

    """

    def __init__(
        self, current_average, new_value, rel_L1_change, rel_L2_change, num_iter_for_avg, data
    ):
        super().__init__(current_average, new_value, rel_L1_change, rel_L2_change)
        self.num_iter_for_avg = num_iter_for_avg
        self.data = data

    @classmethod
    def from_config_create_iterative_averaging(cls, config, section_name=None):
        """
            Build a moving averaging object from config
        Args:
            config (dict): Configuration dict
            section_name (str): Name of section where the averaging object is configured

        Returns:
            MovingAveraging object

        """
        current_average = None
        new_value = None
        rel_L1_change = 1
        rel_L2_change = 1
        if section_name:
            num_iter_for_avg = config[section_name].get("num_iter_for_avg")
        else:
            num_iter_for_avg = config.get("num_iter_for_avg")
        data = []
        return cls(current_average, new_value, rel_L1_change, rel_L2_change, num_iter_for_avg, data)

    def average_computation(self, new_value):
        """Compute the moving average.

        Returns:
            average (np.array): The current average
        """
        self.data.append(new_value.copy())
        if len(self.data) > self.num_iter_for_avg:
            self.data = self.data[-self.num_iter_for_avg :]
        average = 0
        for d in self.data:
            average += d
        return average / len(self.data)


class PolyakAveraging(IterativeAveraging):
    """
    Polyak averaging:
        :math:`x^{(j)}_{avg}=\\frac{1}{j}\\sum_{i=0}^{j}x^{(j)}`

    Attributes:
        current_average (np.array): Current average value
        new_value (np.array): New value for the averaging process
        rel_L1_change (float): Relative change in L1 norm of the average value
        rel_L2_change (float): Relative change in L2 norm of the average value
        iteration_counter (float): Number of samples
        sum_over_iter (np.array): Sum over all samples

    """

    def __init__(
        self,
        current_average,
        new_value,
        rel_L1_change,
        rel_L2_change,
        iteration_counter,
        sum_over_iter,
    ):
        super().__init__(current_average, new_value, rel_L1_change, rel_L2_change)
        self.iteration_counter = iteration_counter
        self.sum_over_iter = sum_over_iter

    @classmethod
    def from_config_create_iterative_averaging(cls, config, section_name=None):
        """
            Build a Polyak averaging object from config
        Args:
            config (dict): Configuration dict
            section_name (str): Name of section where the averaging object is created

        Returns:
            PolyakAveraging object

        """
        current_average = None
        new_value = None
        rel_L1_change = 1
        rel_L2_change = 1
        # Start counter at 1 as in the first update the counter is not increased
        iteration_counter = 1
        sum_over_iter = 0
        return cls(
            current_average,
            new_value,
            rel_L1_change,
            rel_L2_change,
            iteration_counter,
            sum_over_iter,
        )

    def average_computation(self, new_value):
        """Compute the Polyak average.

        Returns:
            current_average (np.array): returns the current average
        """

        self.sum_over_iter += new_value
        self.iteration_counter += 1
        current_average = self.sum_over_iter / self.iteration_counter

        return current_average


class ExponentialAveraging(IterativeAveraging):
    """
    Exponential averaging:
        :math:`x^{(0)}_{avg}=x^{(0)}`
        :math:`x^{(j)}_{avg}= \\alpha x^{(j-1)}_{avg}+(1-\\alpha)x^{(j)}`

        Is also sometimes referred to as exponential smoothing.

    Args:
        current_average (np.array): Current average value
        new_value (np.array): New value for the averaging process
        rel_L1_change (float): Relative change in L1 norm of the average value
        rel_L2_change (float): Relative change in L2 norm of the average value
        coefficient (float): Coefficient in (0,1) for the average

    """

    def __init__(self, current_average, new_value, rel_L1_change, rel_L2_change, coefficient):
        super().__init__(current_average, new_value, rel_L1_change, rel_L2_change)
        self.coefficient = coefficient

    @classmethod
    def from_config_create_iterative_averaging(cls, config, section_name=None):
        """
            Build a exponential averaging object from config
        Args:
            config (dict): Configuration dict
            section_name (str): Name of section where the averaging object is created

        Returns:
            ExponentialAveraging object

        """
        current_average = None
        new_value = None
        rel_L1_change = 1
        rel_L2_change = 1

        if section_name:
            coefficient = config[section_name].get("coefficient")
        else:
            coefficient = config.get("coefficient")

        if coefficient < 0 or coefficient > 1:
            raise ValueError(f"Coefficient for exponential averaging needs to be in (0,1)")
        return cls(current_average, new_value, rel_L2_change, rel_L2_change, coefficient)

    def average_computation(self, new_value):
        """Compute the exponential average.

        Returns:
            current_average (np.array): returns the current average
        """
        current_average = (
            self.coefficient * self.current_average + (1 - self.coefficient) * new_value
        )
        return current_average


def L1_norm(x, averaged=False):
    """Compute the L1 norm of the vector `x`.

    Args:
        x (np.array): Vector
        averaged (bool): If enabled the norm is divided by the number of components

    Returns:
        norm (float): L1 norm of `x`
    """
    x = np.array(x).flatten()
    x = np.nan_to_num(x)
    norm_x = np.sum(np.abs(x))
    if averaged:
        norm_x /= len(x)
    return norm_x


def L2_norm(x, averaged=False):
    """Compute the L2 norm of the vector `x`.

    Args:
        x (np.array): Vector
        averaged (bool): If enabled the norm is divided by the square root of the number of
                         components

    Returns:
        norm (float): L2 norm of `x`
    """
    x = np.array(x).flatten()
    x = np.nan_to_num(x)
    norm_x = np.sum(x**2) ** 0.5
    if averaged:
        norm_x /= len(x) ** 0.5
    return norm_x


def relative_change(old_value, new_value, norm):
    """Compute the relative change of the old and new value for a given norm.

    Args:
        old_value (np.array): Old values
        new_value (np.array): New values
        norm (func): Function to compute a norm

    Returns:
        Relative change
    """
    increment = old_value - new_value
    increment = np.nan_to_num(increment)
    return norm(increment) / (norm(old_value) + 1e-16)
