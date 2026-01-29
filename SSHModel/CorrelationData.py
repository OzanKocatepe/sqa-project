from dataclasses import dataclass, field
import numpy as np
from Fourier import Fourier

@dataclass
class CorrelationData:
    """Contains all of the data relating to the single- and double-time correlations."""
    singleTime: np.ndarray[complex]
    doubleTime: np.ndarray[complex]
    tauAxisSec: np.ndarray[float]
    tauAxisDim: np.ndarray[float]
    tAxisSec: np.ndarray[float]
    tAxisDim: np.ndarray[float]
    singleTimeFourier: list[Fourier] = field(default_factory = list)

    def __init__(self, singleTime: np.ndarray[complex]=None, doubleTime: np.ndarray[complex]=None, tauAxisSec: np.ndarray[float]=None, tauAxisDim: np.ndarray[float]=None, tAxisSec: np.ndarray[float]=None, tAxisDim: np.ndarray[float]=None):
        r"""
        Initialises a CorrelationData instance. Can be instantiated with only some of the data upon creation.
        
        Parameters
        ----------
        singleTime : ndarray[complex]
            The single-time correlation data, with shape (3, t) where t is the number of points in time that we evaluate at.
        doubleTime : ndarray[complex]
            The double-time correlation data, with shape (3, 3, m, t) where the first two dimensions correspond to which two operators we are considering, the third dimension corresponds to the value of the time t, and the last dimension corresponds to the value of the time offset tau.
        tauAxisSec : ndarray[float]
            The values of the time offset, in seconds, that we consider. These are the values of $\tau$ for double-time correlations, and the values of t for single-time correlations (since we are considering an offset from t = 0).
        tauAxisDim : ndarray[float]
            The same as tauAxisSec, but in dimensionless units of $\gamma_-^{-1}$.
        tAxisSec : ndarray[float]
            The values of t, in seconds, that we consider. These are the initial conditions of our double-time correlations when
            $\tau = 0$.
        tAxisDim : ndarray[float]
            The same as tAxisSec, but in dimensionless units of $\gamma_-^{-1}$.
        """

        self.singleTime = singleTime
        self.doubleTime = doubleTime
        self.tauAxisSec = tauAxisSec
        self.tauAxisDim = tauAxisDim
        self.tAxisSec = tAxisSec
        self.tAxisDim = tAxisDim