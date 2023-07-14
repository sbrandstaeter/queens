"""Stochastic optimizers."""
import abc
import logging

import numpy as np

from pqueens.utils.iterative_averaging_utils import (
    ExponentialAveraging,
    l1_norm,
    l2_norm,
    relative_change,
)
from pqueens.utils.print_utils import get_str_table
from pqueens.utils.valid_options_utils import get_option

_logger = logging.getLogger(__name__)


def from_config_create_optimizer(config, optimizer_name=None):
    """Create an optimizer object from dict.

    Args:
        config (dict): Configuration dict
        optimizer_name (str): Name of the stochastic optimizer

    Returns:
        StochasticOptimizer object
    """
    valid_types = {"adam": Adam, "rms_prop": RMSprop, "adamax": Adamax}
    optimizer_type = config.get(optimizer_name)["type"]
    optimizer_class = get_option(
        valid_types, optimizer_type, error_message="Unknown stochastic optimizer."
    )
    optimizer_options = config[optimizer_name].copy()
    optimizer_options.pop('type')
    optimizer = optimizer_class(**optimizer_options)
    return optimizer


class StochasticOptimizer(metaclass=abc.ABCMeta):
    """Base class for stochastic optimizers.

    The optimizers are implemented as generators. This increases the modularity of this class,
    since an object can be used in different settings. Some examples:

    - **Example 1:** Simple optimization run (does not strongly benefit from its generator nature):
        1. Define a gradient function *gradient()*
        2. Create a optimizer object *optimizer* with the gradient function *gradient*
        3. Run the optimization by *optimizer.run_optimization()* in your script

    - **Example 2:** Adding additional functionality during the optimization
        1. Define a optimizer object using a gradient function.
        2. Example code snippet:

            for parameters in optimizer:

                rel_l2_change_params=optimizer.rel_l2_change

                iteration=optimizer.iteration

                # Verbose output

                print(f"Iter {iteration}, parameters {parameters}, rel L2 change "

                f"{rel_l2_change:.2f}")

                # Some additional condition to stop optimization

                if self.number_of_simulations >= 1000:

                    break

    - **Example 3:** Running multiple optimizer iteratively sequentially:
        1. Define *optimizer1* and *optimizer2* with different gradient functions
        2. Example code:

            while not done_bool:

                if not optimizer1.done:

                    self.parameters1=next(optimizer1)

                if not optimizer2.done:

                    self.parameters2=next(optimizer2)

                # Example on how to reduce the learning rate for optimizer2

                if optimizer2.iteration % 1000 == 0:

                    optimizer2.learning_rate *= 0.5

                done_bool = optimizer1.done and optimizer2.done

    Attributes:
        learning_rate (float): Learning rate for the optimizer.
        clip_by_l2_norm_threshold (float): Threshold to clip the gradient by L2-norm.
        clip_by_value_threshold (float): Threshold to clip the gradient components.
        max_iteration (int): Maximum number of iterations.
        precoefficient (int): Is 1 in case of maximization and -1 for minimization.
        rel_l1_change_threshold (float): If the L1 relative change in parameters falls below this
                                         value, this criteria catches.
        rel_l2_change_threshold (float): If the L2 relative change in parameters falls below this
                                         value, this criteria catches.
        iteration (int): Number of iterations done in the optimization so far.
        done (bool): True if the optimization is done.
        rel_l2_change (float): Relative change in L2-norm of variational params w.r.t. the previous
                              iteration.
        rel_l1_change (float): Relative change in L1-norm of variational params w.r.t. the previous
                              iteration.
        current_variational_parameters (np.array): Variational parameters.
        current_gradient_value (np.array): Current gradient vector w.r.t. the variational
                                           parameters.
        gradient (function): Function to compute the gradient.
    """

    _name = "Stochastic Optimizer"

    def __init__(
        self,
        learning_rate,
        optimization_type,
        rel_l1_change_threshold=None,
        rel_l2_change_threshold=None,
        clip_by_l2_norm_threshold=1e6,
        clip_by_value_threshold=1e6,
        max_iter=1e6,
    ):
        """Initialize stochastic optimizer.

        Args:
            learning_rate (float): Learning rate for the optimizer
            optimization_type (str): "max" in case of maximization and "min" for minimization
            rel_l1_change_threshold (float): If the L1 relative change in parameters falls below
                                             this value, this criteria catches.
            rel_l2_change_threshold (float): If the L2 relative change in parameters falls below
                                             this value, this criteria catches.
            clip_by_l2_norm_threshold (float): Threshold to clip the gradient by L2-norm
            clip_by_value_threshold (float): Threshold to clip the gradient components
            max_iter (int): Maximum number of iterations
        """
        valid_options = {"min": -1, "max": 1}
        self.precoefficient = get_option(
            valid_options, optimization_type, error_message="Unknown optimization type."
        )
        self.learning_rate = learning_rate
        self.clip_by_l2_norm_threshold = clip_by_l2_norm_threshold
        self.clip_by_value_threshold = clip_by_value_threshold
        self.max_iteration = max_iter
        self.rel_l2_change_threshold = rel_l2_change_threshold
        self.rel_l1_change_threshold = rel_l1_change_threshold
        self.iteration = 0
        self.done = False
        self.rel_l2_change = None
        self.rel_l1_change = None
        self.current_variational_parameters = None
        self.current_gradient_value = None
        self.gradient = None

    @classmethod
    def from_config_create_optimizer(cls, config, optimizer_name):
        """Create a StochasticOptimizer object from dict.

        Args:
            config (dict): Configuration dict
            optimizer_name (str): Name of section in which optimizer is defined
        """
        optimizer_options = config[optimizer_name].copy()
        optimizer_options.pop('type')
        return cls(**optimizer_options)

    @abc.abstractmethod
    def scheme_specific_gradient(self, gradient):
        """Scheme specific gradient computation.

        Here the gradient is transformed according to the desired stochastic optimization approach.

        Args:
            gradient (np.array): Current gradient
        """

    def set_gradient_function(self, gradient_function):
        """Set the gradient function.

        The *gradient_function* has to be a function of the parameters and returns
        the gradient value.

        Args:
            gradient_function (function): Gradient function.
        """
        self.gradient = gradient_function

    def _compute_rel_change(self, old_parameters, new_parameters):
        """Compute L1 and L2 based relative changes of variational parameters.

        Args:
            old_parameters (np.array): Old parameters
            new_parameters (np.array): New parameters
        """
        self.rel_l2_change = relative_change(
            old_parameters, new_parameters, lambda x: l2_norm(x, averaged=True)
        )
        self.rel_l1_change = relative_change(
            old_parameters, new_parameters, lambda x: l1_norm(x, averaged=True)
        )

    def do_single_iteration(self, gradient):
        r"""Single iteration for a given gradient.

        Iteration step for a given gradient :math:`g`:
            :math:`p^{(i+1)}=p^{(i)}+\beta \alpha g`
        where :math:`\beta=-1` for minimization and +1 for maximization and :math:`\alpha` is
        the learning rate.

        Args:
            gradient (np.array): Current gradient
        """
        self.current_variational_parameters = (
            self.current_variational_parameters
            + self.precoefficient * self.learning_rate * gradient
        )
        self.iteration += 1

    def clip_gradient(self, gradient):
        """Clip the gradient by value and then by norm.

        Args:
            gradient (np.array): Current gradient

        Returns:
            gradient (np.array): The clipped gradient
        """
        gradient = clip_by_value(gradient, self.clip_by_value_threshold)
        gradient = clip_by_l2_norm(gradient, self.clip_by_l2_norm_threshold)
        return gradient

    def __next__(self):
        """Python intern function to make this object a generator.

        Essentially this is a single iteration of the stochastic optimizer consiting of:

            1. Computing the noisy gradient
            2. Clipping the gradient
            3. Transform the gradient using the scheme specific approach
            4. Update the parameters
            5. Compute relative changes
            6. Check if optimization is done

        Returns:
            current_variational_parameters (np.array): current variational parameters of the
            optimization
        """
        if self.done:
            raise StopIteration
        old_parameters = self.current_variational_parameters.copy()
        current_gradient = self.gradient(self.current_variational_parameters)
        current_gradient = self.clip_gradient(current_gradient)
        current_gradient = self.scheme_specific_gradient(current_gradient)
        self.current_gradient_value = current_gradient
        self.do_single_iteration(current_gradient)
        self._compute_rel_change(old_parameters, self.current_variational_parameters)
        self._check_if_done()
        return self.current_variational_parameters

    def __iter__(self):
        """Python intern function needed to make this object iterable.

        Hence it can be called as:
            for p in optimizer:
                print(f"Current parameters: {p}")

        Returns:
            self
        """
        return self

    def _check_if_done(self):
        """Check if optimization is done based on L1 and L2 norms.

        Criteria are based on the change of the variational parameters.
        """
        if np.any(np.isnan(self.current_variational_parameters)):
            raise ValueError("At least one of the variational parameters is NaN")
        self.done = (
            self.rel_l2_change <= self.rel_l2_change_threshold
            and self.rel_l1_change <= self.rel_l1_change_threshold
        ) or self.iteration >= self.max_iteration

    def run_optimization(self):
        """Run the optimization.

        Returns:
            np.array: variational parameters
        """
        for _ in self:
            pass
        return self.current_variational_parameters

    def _get_print_dict(self):
        """Get print dict.

        Returns:
            dict: dictionary with data to print
        """
        if self.precoefficient == 1:
            optimization_type = "maximization"
        else:
            optimization_type = "minimization"

        print_dict = {
            "Optimization type": optimization_type,
            "Learning_rate": self.learning_rate,
            "Iterations": self.iteration,
            "Max. number of iteration": self.max_iteration,
            "Rel. L1 change of the parameters": self.rel_l1_change,
            "Rel. L1 change threshold": self.rel_l1_change_threshold,
            "Rel. L2 change of the parameters": self.rel_l2_change,
            "Rel. L2 change threshold": self.rel_l2_change_threshold,
        }
        return print_dict

    def __str__(self):
        """String of optimizer.

        Returns:
            str: String version of the optimizer
        """
        print_dict = self._get_print_dict()
        return get_str_table(self._name, print_dict)


class RMSprop(StochasticOptimizer):
    r"""RMSprop stochastic optimizer [1].

    References:
        [1] Tieleman and Hinton. "Lecture 6.5-rmsprop: Divide the gradient by a running average of
            its recent magnitude". Coursera. 2012.

    Attributes:
        beta (float):  :math:`\beta` parameter as described in [1].
        v (ExponentialAveragingObject): Exponential average of the gradient momentum.
        eps (float): Nugget term to avoid a division by values close to zero.
    """

    _name = "RMSprop Stochastic Optimizer"

    def __init__(
        self,
        learning_rate,
        optimization_type,
        rel_l1_change_threshold=None,
        rel_l2_change_threshold=None,
        clip_by_l2_norm_threshold=1e6,
        clip_by_value_threshold=1e6,
        max_iter=1e6,
        beta=0.999,
        eps=1e-8,
    ):
        """Initialize optimizer.

        Args:
            learning_rate (float): Learning rate for the optimizer
            optimization_type (str): "max" in case of maximization and "min" for minimization
            rel_l1_change_threshold (float): If the L1 relative change in parameters falls below
                                             this value, this criteria catches.
            rel_l2_change_threshold (float): If the L2 relative change in parameters falls below
                                            this value, this criteria catches.
            clip_by_l2_norm_threshold (float): Threshold to clip the gradient by L2-norm
            clip_by_value_threshold (float): Threshold to clip the gradient components
            max_iter (int): Maximum number of iterations
            beta (float): :math:`beta` parameter as described in [1]
            eps (float): Nugget term to avoid a division by values close to zero
        """
        super().__init__(
            learning_rate=learning_rate,
            optimization_type=optimization_type,
            rel_l1_change_threshold=rel_l1_change_threshold,
            rel_l2_change_threshold=rel_l2_change_threshold,
            clip_by_l2_norm_threshold=clip_by_l2_norm_threshold,
            clip_by_value_threshold=clip_by_value_threshold,
            max_iter=max_iter,
        )
        self.beta = beta
        self.v = ExponentialAveraging(coefficient=beta)  # pylint: disable=invalid-name
        self.eps = eps

    def scheme_specific_gradient(self, gradient):
        """Rmsprop gradient computation.

        Args:
            gradient (np.array): Gradient

        Returns:
            gradient (np.array): RMSprop gradient
        """
        if self.iteration == 0:
            self.v.current_average = np.zeros(gradient.shape)

        v_hat = self.v.update_average(gradient**2)
        v_hat /= 1 - self.beta ** (self.iteration + 1)
        gradient = gradient / (v_hat**0.5 + self.eps)
        return gradient


class Adam(StochasticOptimizer):
    r"""Adam stochastic optimizer [1].

    References:
        [1] Kingma and Ba. "Adam: A Method for Stochastic Optimization".  ICLR 2015. 2015.

    Attributes:
        beta_1 (float):  :math:`\beta_1` parameter as described in [1].
        beta_2 (float):  :math:`\beta_2` parameter as described in [1].
        m (ExponentialAveragingObject): Exponential average of the gradient.
        v (ExponentialAveragingObject): Exponential average of the gradient momentum.
        eps (float): Nugget term to avoid a division by values close to zero.
    """

    _name = "Adam Stochastic Optimizer"

    def __init__(
        self,
        learning_rate,
        optimization_type,
        rel_l1_change_threshold=None,
        rel_l2_change_threshold=None,
        clip_by_l2_norm_threshold=1e6,
        clip_by_value_threshold=1e6,
        max_iter=1e6,
        beta_1=0.9,
        beta_2=0.999,
        eps=1e-8,
    ):
        """Initialize optimizer.

        Args:
            learning_rate (float): Learning rate for the optimizer
            optimization_type (str): "max" in case of maximization and "min" for minimization
            rel_l1_change_threshold (float): If the L1 relative change in parameters falls below
                                             this value, this criteria catches.
            rel_l2_change_threshold (float): If the L2 relative change in parameters falls below
                                            this value, this criteria catches.
            clip_by_l2_norm_threshold (float): Threshold to clip the gradient by L2-norm
            clip_by_value_threshold (float): Threshold to clip the gradient components
            max_iter (int): Maximum number of iterations
            beta_1 (float): :math:`beta_1` parameter as described in [1]
            beta_2 (float): :math:`beta_1` parameter as described in [1]
            eps (float): Nugget term to avoid a division by values close to zero
        """
        super().__init__(
            learning_rate=learning_rate,
            optimization_type=optimization_type,
            rel_l1_change_threshold=rel_l1_change_threshold,
            rel_l2_change_threshold=rel_l2_change_threshold,
            clip_by_l2_norm_threshold=clip_by_l2_norm_threshold,
            clip_by_value_threshold=clip_by_value_threshold,
            max_iter=max_iter,
        )
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.m = ExponentialAveraging(coefficient=beta_1)  # pylint: disable=invalid-name
        self.v = ExponentialAveraging(coefficient=beta_2)  # pylint: disable=invalid-name
        self.eps = eps

    def scheme_specific_gradient(self, gradient):
        """Adam gradient computation.

        Args:
            gradient (np.array): Gradient

        Returns:
            gradient (np.array): Adam gradient
        """
        if self.iteration == 0:
            self.m.current_average = np.zeros(gradient.shape)
            self.v.current_average = np.zeros(gradient.shape)

        m_hat = self.m.update_average(gradient)
        v_hat = self.v.update_average(gradient**2)
        m_hat /= 1 - self.beta_1 ** (self.iteration + 1)
        v_hat /= 1 - self.beta_2 ** (self.iteration + 1)
        gradient = m_hat / (v_hat**0.5 + self.eps)
        return gradient


class Adamax(StochasticOptimizer):
    r"""Adamax stochastic optimizer [1]. *eps* added to avoid division by zero.

    References:
        [1] Kingma and Ba. "Adam: A Method for Stochastic Optimization".  ICLR 2015. 2015.

    Attributes:
        beta_1 (float): :math:`\beta_1` parameter as described in [1].
        beta_2 (float): :math:`\beta_2` parameter as described in [1].
        m (ExponentialAveragingObject): Exponential average of the gradient.
        u (np.array): Maximum gradient momentum.
        eps (float): Nugget term to avoid a division by values close to zero.
    """

    _name = "Adamax Stochastic Optimizer"

    def __init__(
        self,
        learning_rate,
        optimization_type,
        rel_l1_change_threshold=None,
        rel_l2_change_threshold=None,
        clip_by_l2_norm_threshold=1e6,
        clip_by_value_threshold=1e6,
        max_iter=1e6,
        beta_1=0.9,
        beta_2=0.999,
        eps=1e-8,
    ):
        """Initialize optimizer.

        Args:
            learning_rate (float): Learning rate for the optimizer
            optimization_type (str): "max" in case of maximization and "min" for minimization
            rel_l1_change_threshold (float): If the L1 relative change in parameters falls below
                                             this value, this criteria catches.
            rel_l2_change_threshold (float): If the L2 relative change in parameters falls below
                                            this value, this criteria catches.
            clip_by_l2_norm_threshold (float): Threshold to clip the gradient by L2-norm
            clip_by_value_threshold (float): Threshold to clip the gradient components
            max_iter (int): Maximum number of iterations
            beta_1 (float): :math:`beta_1` parameter as described in [1]
            beta_2 (float): :math:`beta_1` parameter as described in [1]
            eps (float): Nugget term to avoid a division by values close to zero
        """
        super().__init__(
            learning_rate=learning_rate,
            optimization_type=optimization_type,
            rel_l1_change_threshold=rel_l1_change_threshold,
            rel_l2_change_threshold=rel_l2_change_threshold,
            clip_by_l2_norm_threshold=clip_by_l2_norm_threshold,
            clip_by_value_threshold=clip_by_value_threshold,
            max_iter=max_iter,
        )
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.m = ExponentialAveraging(coefficient=beta_1)  # pylint: disable=invalid-name
        self.u = 0  # pylint: disable=invalid-name
        self.eps = eps

    def scheme_specific_gradient(self, gradient):
        """Adamax gradient computation.

        Args:
            gradient (np.array): Gradient

        Returns:
            gradient (np.array): Adam gradient
        """
        if self.iteration == 0:
            self.m.current_average = np.zeros(gradient.shape)
            self.u = np.zeros(gradient.shape)

        m_hat = self.m.update_average(gradient)
        m_hat /= 1 - self.beta_1 ** (self.iteration + 1)
        abs_grad = np.abs(gradient)
        self.u = np.maximum(self.beta_2 * self.u, abs_grad)
        gradient = m_hat / (self.u + self.eps)
        return gradient


def clip_by_l2_norm(gradient, l2_norm_threshold=1e6):
    """Clip gradients by L2-norm.

    Args:
        gradient (np.array): Gradient
        l2_norm_threshold (float): Clipping threshold
    Returns:
        gradient (np.array): Clipped gradients
    """
    gradient = np.nan_to_num(gradient)
    gradient_l2_norm = l2_norm(gradient)
    if gradient_l2_norm > l2_norm_threshold:
        gradient /= gradient_l2_norm / l2_norm_threshold
        _logger.warning("Gradient clipped due to large norm!")
    return gradient


def clip_by_value(gradient, threshold=1e6):
    """Clip gradients by value.

    Clips if the absolute value op the component is larger than the threshold.

    Args:
        gradient (np.array): Gradient
        threshold (float): Threshold to clip
    Returns:
        gradient (np.array): Clipped gradients
    """
    gradient = np.nan_to_num(gradient)
    gradient = np.clip(gradient, -threshold, threshold)
    return gradient
