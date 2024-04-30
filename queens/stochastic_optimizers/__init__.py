"""Stochastic optimizers init."""

from queens.stochastic_optimizers.adam import Adam
from queens.stochastic_optimizers.adamax import Adamax
from queens.stochastic_optimizers.rms_prop import RMSprop
from queens.stochastic_optimizers.sgd import SGD

VALID_TYPES = {"adam": Adam, "adamax": Adamax, "rms_prop": RMSprop, "sgd": SGD}
