from dataclasses import dataclass
import numpy as np

@dataclass
class CorrelationData:
    """Contains all of the data relating to the single- and double-time correlations."""
    singleTime: np.ndarray[complex]
    doubleTime: np.ndarray[complex]
    tauAxisSec: np.ndarray[float]
    tauAxisDim: np.ndarray[float]
    tAxisDim: np.ndarray[float]
    tAxisSec: np.ndarray[float]