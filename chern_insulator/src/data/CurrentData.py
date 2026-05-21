from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

@dataclass(slots=True)
class CurrentData:
    """
    Stores the current data for a model.

    Parameters
    ----------
    paramagneticCurrent : ndarray[complex]
        Stores the paramagnetic current in an array of shape (2, time.size), where
        the first dimension differentiates between the x-component (index 0) and
        the y-component (index 1) of the current operator.
    diamagneticCurrent : ndarray[complex]
        Stores the xx-diamagnetic current in an array of shape (time.size,).
    totalCurrent : ndarray[complex]
        The total current, calculated from diamagnetic and paramagnetic currents,
        with shape (2, time.size).
    doubleTimeCurrent : ndarray[complex]
        Stores the double-time currents in an array of shape (2, 2, t.size, tau.size),
        with the first two axes corresponding to the left and right current direction respectively.
    lengthGaugeCurrent : ndarray[complex]
        The first-order current calculated in the length gauge.
    meanSecondOrderCurrent : ndarray[complex]
        The double-time current integrated along the t axis, with shape (2, 2, tau.size).
    spectralNoiseTensor : ndarray[complex]
        The fourier transforms at the harmonics of the driving frequency of the second order current
        (correlation tensor) at each time t.
        Of shape (2, 2, 2 * maxN + 1, t.size), where the first two axes correspond to the subscripts of
        the correlation tensor, the third axis corresponds to the harmonics, from -maxN to maxN,
        and the last axis corresponds to the time t. This is a function of t.
    """

    paramagneticCurrent: np.ndarray[complex] = field(default = None)
    diamagneticCurrent: np.ndarray[complex] = field(default = None)
    totalCurrent: np.ndarray[complex] = field(default = None)
    doubleTimeCurrent: np.ndarray[complex] = field(default = None)
    lengthGaugeCurrent: np.ndarray[complex] = field(default = None)
    meanSecondOrderCurrent: np.ndarray[complex] = field(default = None)
    spectralNoiseTensor: np.ndarray[complex] = field(default = None)

    def __add__(self, other: CurrentData) -> CurrentData:
        """
        Adds together two CurrentData instances by adding together each attribute.

        Parameters
        ----------
        other : CurrentData
            The other CurrentData instance to add to this one.

        Returns
        -------
        CurrentData:
            A new CurrentData instance that has attributes equal to the sum
            of the attributes of the two operands.
        """

        return CurrentData(
            paramagneticCurrent = self.paramagneticCurrent + other.paramagneticCurrent,
            diamagneticCurrent = self.diamagneticCurrent + other.diamagneticCurrent,
            totalCurrent = self.totalCurrent + other.totalCurrent,
            doubleTimeCurrent = self.doubleTimeCurrent + other.doubleTimeCurrent
                if self.doubleTimeCurrent is not None and other.doubleTimeCurrent is not None
                else None,
            lengthGaugeCurrent = self.lengthGaugeCurrent + other.lengthGaugeCurrent
                if self.lengthGaugeCurrent is not None and other.lengthGaugeCurrent is not None
                else None,
            meanSecondOrderCurrent = self.meanSecondOrderCurrent + other.meanSecondOrderCurrent,
            spectralNoiseTensor = self.spectralNoiseTensor + other.spectralNoiseTensor
        )
    
    def __truediv__(self, other: int) -> CurrentData:
        """Divides the instance by an int.

        We do not require other instances of division in this code currently.
        
        Parameters
        ----------
        other : int
            The integer to divide by.
            
        Returns
        -------
        CurrentData
            A new instance with each component divided by the int.
        """
        
        # No checks since we only divide in one place, and if we need to do another division
        # I trust myself to know that it passes directly to the inner arrays.
        # Could add checks later.
        return CurrentData(
            paramagneticCurrent = self.paramagneticCurrent / other,
            diamagneticCurrent = self.diamagneticCurrent / other,
            totalCurrent = self.totalCurrent / other,
            doubleTimeCurrent = self.doubleTimeCurrent / other
                if self.doubleTimeCurrent is not None
                else None,
            lengthGaugeCurrent = self.lengthGaugeCurrent / other
                if self.lengthGaugeCurrent is not None
                else None,
            meanSecondOrderCurrent = self.meanSecondOrderCurrent / other,
            spectralNoiseTensor = self.spectralNoiseTensor / other
        )