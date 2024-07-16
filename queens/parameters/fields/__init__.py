"""Random Fields."""

from queens.parameters.fields.fourier_field import FourierRandomField
from queens.parameters.fields.kl_field import KarhunenLoeveRandomField
from queens.parameters.fields.piece_wise_field import PieceWiseRandomField

VALID_TYPES = {
    "kl": KarhunenLoeveRandomField,
    "fourier": FourierRandomField,
    "piece-wise": PieceWiseRandomField,
}
