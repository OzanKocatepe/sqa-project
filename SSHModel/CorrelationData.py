from dataclasses import dataclass, field
import numpy as np
from Fourier import Fourier

@dataclass
class CorrelationData:
    """Contains all of the data relating to the single- and double-time correlations."""
    singleTime: np.ndarray[complex] = None
    doubleTime: np.ndarray[complex] = None
    tauAxisSec: np.ndarray[float] = None
    tauAxisDim: np.ndarray[float] = None
    tAxisSec: np.ndarray[float] = None
    tAxisDim: np.ndarray[float] = None
    singleTimeFourier: list[Fourier] = field(default_factory = list)