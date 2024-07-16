"""Variational distributions init."""

from queens.variational_distributions.full_rank_normal import FullRankNormalVariational
from queens.variational_distributions.joint import JointVariational
from queens.variational_distributions.mean_field_normal import MeanFieldNormalVariational
from queens.variational_distributions.mixture_model import MixtureModelVariational
from queens.variational_distributions.particle import ParticleVariational

VALID_TYPES = {
    "mean_field_variational": MeanFieldNormalVariational,
    "full_rank_variational": FullRankNormalVariational,
}
