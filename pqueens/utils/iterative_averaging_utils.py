"""Iterative averaging utils."""
import abc

import numpy as np

from pqueens.utils.print_utils import get_str_table
from pqueens.utils.valid_options_utils import get_option


def from_config_create_iterative_averaging(config):
    """Build an iterative averaging scheme from config.

    Args:
        config (dict): Configuration dict for the iterative averaging

    Returns:
        Iterative averaging object
    """
    valid_options = {
        "moving_average": MovingAveraging,
        "polyak_averaging": PolyakAveraging,
        "exponential_averaging": ExponentialAveraging,
    }
    averaging_type = config.get("averaging_type")

    averaging_class = get_option(
        valid_options, averaging_type, error_message="Iterative averaging option not found."
    )
    return averaging_class.from_config_create_iterative_averaging(config)


class IterativeAveraging(metaclass=abc.ABCMeta):
    """Base class for iterative averaging schemes.

    Attributes:
        current_average (np.array): Current average value.
        new_value (np.array): New value for the averaging process.
        rel_L1_change (float): Relative change in L1 norm of the average value.
        rel_L2_change (float): Relative change in L2 norm of the average value.
    """

    def __init__(self):
        """Initialize iterative averaging."""
        self.current_average = None
        self.new_value = None
        self.rel_L1_change = 1
        self.rel_L2_change = 1

    def update_average(self, new_value):
        """Compute the actual average.

        Args:
            new_value (np.array): New observation for the averaging

        Returns:
            Current average value
        """
        if isinstance(new_value, (float, int)):
            new_value = np.array(new_value)
        if self.current_average is not None:
            old_average = self.current_average.copy()
            self.current_average = self.average_computation(new_value)
            self.rel_L2_change = relative_change(old_average, self.current_average, L2_norm)
            self.rel_L1_change = relative_change(old_average, self.current_average, L1_norm)
        else:
            # If it is the first observation
            self.current_average = new_value.copy()
        return self.current_average.copy()

    def __str__(self, name, approach_print_dict):
        """String of the iterative averaging.

        Args:
            name (str): averaging approach
            approach_print_dict (dict): Dict with method specific print quantities

        Returns:
            str: String version of the optimizer
        """
        print_dict = {
            "Rel. L1 change to previous average": self.rel_L1_change,
            "Rel. L2 change to previous average": self.rel_L2_change,
            "Current average": self.current_average,
        }
        approach_print_dict.update(print_dict)
        return get_str_table(name, approach_print_dict)

    @abc.abstractmethod
    def average_computation(self):
        """Here the averaging approach is implemented."""
        pass


class MovingAveraging(IterativeAveraging):
    r"""Moving averages.

    :math:`x^{(j)}_{avg}=\frac{1}{k}\sum_{i=0}^{k-1}x^{(j-i)}`

    where :math:`k-1` is the number of values from previous iterations that are used

    Attributes:
        num_iter_for_avg (int): Number of samples in the averaging window.
        data: TODO_doc
    """

    def __init__(self, num_iter_for_avg):
        """Initialize moving averaging object.

        Args:
            num_iter_for_avg (int): Number of samples in the averaging window
        """
        super().__init__()
        self.num_iter_for_avg = num_iter_for_avg
        self.data = []

    @classmethod
    def from_config_create_iterative_averaging(cls, config, section_name=None):
        """Build a moving averaging object from config.

        Args:
            config (dict): Configuration dict
            section_name (str): Name of section where the averaging object is configured

        Returns:
            MovingAveraging object
        """
        if section_name:
            num_iter_for_avg = config[section_name].get("num_iter_for_avg")
        else:
            num_iter_for_avg = config.get("num_iter_for_avg")
        return cls(num_iter_for_avg=num_iter_for_avg)

    def average_computation(self, new_value):
        """Compute the moving average.

        Args:
            new_value (float or np.array): New value to update the average

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

    def __str__(self):
        """String of the iterative averaging.

        Returns:
            str: String version of the iterative averaging
        """
        name = "Moving average."
        print_dict = {
            "Averaging window size": self.num_iter_for_avg,
        }
        return super().__str__(name, print_dict)


class PolyakAveraging(IterativeAveraging):
    r"""Polyak averaging.

    :math:`x^{(j)}_{avg}=\frac{1}{j}\sum_{i=0}^{j}x^{(j)}`

    Attributes:
        iteration_counter (float): Number of samples.
        sum_over_iter (np.array): Sum over all samples.

    """

    def __init__(self):
        """Initialize Polyak averaging object."""
        super().__init__()
        self.iteration_counter = 1
        self.sum_over_iter = 0

    @classmethod
    def from_config_create_iterative_averaging(cls, config, section_name=None):
        """Build a Polyak averaging object from config.

        Args:
            config (dict): Configuration dict
            section_name (str): Name of section where the averaging object is created

        Returns:
            PolyakAveraging object
        """
        return cls()

    def average_computation(self, new_value):
        """Compute the Polyak average.

        Args:
            new_value (float or np.array): New value to update the average

        Returns:
            current_average (np.array): Returns the current average
        """
        self.sum_over_iter += new_value
        self.iteration_counter += 1
        current_average = self.sum_over_iter / self.iteration_counter

        return current_average

    def __str__(self):
        """String of the iterative averaging.

        Returns:
            str: String version of the iterative averaging
        """
        name = "Polyak averaging."
        print_dict = {
            "Number of iterations": self.iteration_counter,
        }
        return super().__str__(name, print_dict)


class ExponentialAveraging(IterativeAveraging):
    r"""Exponential averaging.

    :math:`x^{(0)}_{avg}=x^{(0)}`

    :math:`x^{(j)}_{avg}= \alpha x^{(j-1)}_{avg}+(1-\alpha)x^{(j)}`

    Is also sometimes referred to as exponential smoothing.

    Attributes:
        coefficient (float): Coefficient in (0,1) for the average.

    """

    def __init__(self, coefficient):
        """Initialize exponential averaging object.

        Args:
            coefficient (float): Coefficient in (0,1) for the average
        """
        super().__init__()
        self.coefficient = coefficient

    @classmethod
    def from_config_create_iterative_averaging(cls, config, section_name=None):
        """Build a exponential averaging object from config.

        Args:
            config (dict): Configuration dict
            section_name (str): Name of section where the averaging object is created

        Returns:
            ExponentialAveraging object
        """
        if section_name:
            coefficient = config[section_name].get("coefficient")
        else:
            coefficient = config.get("coefficient")

        if coefficient < 0 or coefficient > 1:
            raise ValueError("Coefficient for exponential averaging needs to be in (0,1)")
        return cls(coefficient=coefficient)

    def average_computation(self, new_value):
        """Compute the exponential average.

        Args:
            new_value (float or np.array): New value to update the average.

        Returns:
            current_average (np.array): Returns the current average
        """
        current_average = (
            self.coefficient * self.current_average + (1 - self.coefficient) * new_value
        )
        return current_average

    def __str__(self):
        """String of the iterative averaging.

        Returns:
            str: String version of the iterative averaging
        """
        print_dict = {
            "Coefficient": self.coefficient,
        }
        return super().__str__("Exponential averaging.", print_dict)


def L1_norm(x, averaged=False):
    """Compute the L1 norm of the vector *x*.

    Args:
        x (np.array): Vector
        averaged (bool): If enabled, the norm is divided by the number of components

    Returns:
        norm (float): L1 norm of *x*
    """
    x = np.array(x).flatten()
    x = np.nan_to_num(x)
    norm_x = np.sum(np.abs(x))
    if averaged:
        norm_x /= len(x)
    return norm_x


def L2_norm(x, averaged=False):
    """Compute the L2 norm of the vector *x*.

    Args:
        x (np.array): Vector
        averaged (bool): If enabled the norm is divided by the square root of the number of
                         components

    Returns:
        norm (float): L2 norm of *x*
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
