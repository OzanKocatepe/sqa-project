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
    tAxisDim : ndarray[float]
        The non-dimensional points in time, confined within a single steady-state period,
        which we use as initial conditions for our double-time correlations.
    tAxisSec : ndarray[float]
        Same as tAxisDim, but in units of seconds.
    freqAxis : ndarray[float]
        The axis of the Fourier transform of our operators, calculated based on the
        sampling rate of tauAxis, with units being the multiples of the driving frequency.
    steadyStateCutoff : float
        The non-dimensional point in time that we consider to be the 'steady state' (i.e.
        all transient effects have decayed). This is primarily used to limit the domain
        that we calculate the FFT of our operators on.
    """

    tauAxisDim: np.ndarray[float]
    tauAxisSec: np.ndarray[float]
    tAxisDim: np.ndarray[float]
    tAxisSec: np.ndarray[float]
    freqAxis: np.ndarray[float]
    steadyStateCutoff: float