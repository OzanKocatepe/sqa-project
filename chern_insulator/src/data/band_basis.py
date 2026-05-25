import numpy as np
from dataclasses import dataclass

@dataclass(slots=True, frozen=True)
class BandBasis:
    plusEigenvector: complex
    minusEigenvector: complex
    plusProjector: np.ndarray[complex]
    minusProjector: np.ndarray[complex]