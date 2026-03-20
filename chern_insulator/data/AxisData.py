import numpy as np
from dataclasses import dataclass

@dataclass(slots=True)
class AxisData:
    """Contains all of the relevant data for the axes that we use.
    
    Parameters
    ----------
    tauAxisDim : ndarray[float]
        The non-dimensional points in time to evaluate our operators.
    tauAxisSec : ndarray[float]
        Same as tauAxisDim, but in units of seconds.
    freqAxis : ndarray[float]
        The axis of the Fourier transform of our operators, calculated based on the
        sampling rate of tauAxis, in units of the driving frequency.
    """

    tauAxisDim: np.ndarray[float]
    tauAxisSec: np.ndarray[float]
    freqAxis: np.ndarray[float]