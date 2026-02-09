from dataclasses import dataclass
import numpy as np
from .Fourier import Fourier

@dataclass
class CorrelationData:
    """Contains all of the data relating to the single- and double-time correlations."""
    singleTime: np.ndarray[complex] = None
    doubleTime: np.ndarray[complex] = None
    tauAxisSec: np.ndarray[float] = None
    tauAxisDim: np.ndarray[float] = None
    tAxisSec: np.ndarray[float] = None
    tAxisDim: np.ndarray[float] = None
    singleTimeFourier: list[Fourier] = None

    def __add__(self, other: CorrelationData) -> CorrelationData:
        """
        Adds another CorrelationData instance to itself to create a new CorrelationData instance.
        Assumes that the axes for both instances are the same.

        Parameters
        ----------
        other : CorrelationData
            The other CorrelationData object to add to this one.

        Returns
        -------
        CorrelationData
            The new current data instance, with the same axes as each currentData instance, but
            the sum of their time and fourier data.
        """

        return CorrelationData(
            singleTime = self.singleTime + other.singleTime,
            doubleTime = self.doubleTime + other.doubleTime,
            singleTimeFourier = self.singleTimeFourier + other.singleTimeFourier,
            tauAxisSec = self.tauAxisSec,
            tauAxisDim = self.tauAxisDim,
            tAxisSec = self.tAxisSec,
            tAxisDim = self.tAxisDim
        )