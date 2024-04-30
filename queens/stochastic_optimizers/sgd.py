"""SGD optimizer."""
import logging

from queens.stochastic_optimizers.stochastic_optimizer import StochasticOptimizer

_logger = logging.getLogger(__name__)


class SGD(StochasticOptimizer):
    """Stochastic gradient descent optimizer."""

    _name = "SGD Stochastic Optimizer"

    def __init__(
        self,
        learning_rate,
        optimization_type,
        rel_l1_change_threshold=None,
        rel_l2_change_threshold=None,
        clip_by_l2_norm_threshold=1e6,
        clip_by_value_threshold=1e6,
        max_iteration=1e6,
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
            max_iteration (int): Maximum number of iterations
        """
        super().__init__(
            learning_rate=learning_rate,
            optimization_type=optimization_type,
            rel_l1_change_threshold=rel_l1_change_threshold,
            rel_l2_change_threshold=rel_l2_change_threshold,
            clip_by_l2_norm_threshold=clip_by_l2_norm_threshold,
            clip_by_value_threshold=clip_by_value_threshold,
            max_iteration=max_iteration,
        )

    def scheme_specific_gradient(self, gradient):
        """SGD gradient computation.

        Args:
            gradient (np.array): Gradient

        Returns:
            gradient (np.array): SGD gradient
        """
        return gradient
