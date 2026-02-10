from dataclasses import dataclass
import numpy as np
import scipy.special as special

from .Fourier import Fourier
from .SSHParameters import ModelParameters
from .CorrelationData import CorrelationData

@dataclass
class CurrentData:
    """Contains all of the data relating to the current operator."""

    # The expectation of the current operator in the time- and frequency- domains.
    timeDomainCurrent: np.ndarray[complex] = None
    freqDomainCurrent: np.ndarray[complex] = None

    # The fourier series of the time-domain current.
    currentFourierSeries: Fourier = None

    # The expectation of the double-time current operator.
    doubleTimeCurrent: np.ndarray[complex] = None
    # The expectation of the double-time current operator, integrated w.r.t. t.
    integratedDoubleTimeCurrent: np.ndarray[complex] = None
    # The integrated product of the expectation of the current operator, evaluated at two different times.
    doubleTimeCurrentProduct: np.ndarray[complex] = None

    # The connected correlator for the double-time current operator in time- and frequency- domains.
    timeConnectedCorrelator: np.ndarray[complex] = None
    freqConnectedCorrelator: np.ndarray[complex] = None

    # The fourier transform of the connected-correlator evaluated at integer harmonics of the
    # driving frequency.
    harmonics : np.ndarray[complex] = None

    # The double-time current product (second half of the connected correlator) evaluated
    # numerically using the timeDomainCurrent at multiple times, rather than using the fourier series
    # derived analytically. Used to make the product term in the connected correlator is correct.
    _numericalDoubleTimeCurrentProduct: np.ndarray[float] = None

    def __add__(self, other: CurrentData) -> CurrentData:
        """
        Adds another CurrentData instance to itself to create a new CurrentData instance.

        Parameters
        ----------
        other : CurrentData
            The other CurrentData object to add to this one.

        Returns
        -------
        CurrentData
            The new current data instance with the sum of their time and frequency data.
        """

        return CurrentData(
            timeDomainCurrent = self.timeDomainCurrent + other.timeDomainCurrent,
            freqDomainCurrent = self.freqDomainCurrent + other.freqDomainCurrent,
            currentFourierSeries = self.currentFourierSeries + other.currentFourierSeries,
            doubleTimeCurrent = self.doubleTimeCurrent + other.doubleTimeCurrent,
            integratedDoubleTimeCurrent = self.integratedDoubleTimeCurrent + other.integratedDoubleTimeCurrent,
            doubleTimeCurrentProduct = self.doubleTimeCurrentProduct + other.doubleTimeCurrentProduct,
            timeConnectedCorrelator = self.timeConnectedCorrelator + other.timeConnectedCorrelator,
            freqConnectedCorrelator = self.freqConnectedCorrelator + other.freqConnectedCorrelator,
            harmonics = self.harmonics + other.harmonics,
            _numericalDoubleTimeCurrentProduct = self._numericalDoubleTimeCurrentProduct + other._numericalDoubleTimeCurrentProduct
        )