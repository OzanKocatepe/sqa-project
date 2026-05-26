import numpy as np
from dataclasses import dataclass

@dataclass(slots=True, frozen=True)
class BandBasis:
    plus_eigenvector: complex
    minus_eigenvector: complex
    plus_projection: np.ndarray[complex]
    minus_projection: np.ndarray[complex]