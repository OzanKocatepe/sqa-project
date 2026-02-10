import numpy as np
from dataclasses import dataclass

@dataclass
class AxisData:
    """Contains all of the relevant data for the axes that we use."""

    tauAxisDim: np.ndarray[float]
    tauAxisSec: np.ndarray[float]
    tAxisDim: np.ndarray[float]
    tAxisSec: np.ndarray[float]
    freqAxis: np.ndarray[float]