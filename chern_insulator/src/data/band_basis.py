import numpy as np
from dataclasses import dataclass

@dataclass(slots=True, frozen=True)
class BandBasis:
    plusEigenvector: complex
    minusEigenvector: complex
    plusProjection: np.ndarray[complex]
    minusProjection: np.ndarray[complex]