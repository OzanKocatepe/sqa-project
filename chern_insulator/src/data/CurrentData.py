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
        Stores the diamagnetic current in an array of shape (time.size).
    """

    paramagneticCurrent: np.ndarray[complex] = field(default = None)
    diamagneticCurrent: np.ndarray[complex] = field(default = None)
    totalCurrent: np.ndarray[complex] = field(default = None)
    doubleTimeCurrent: np.ndarray[complex] = field(default = None)
    lengthGaugeCurrent: np.ndarray[complex] = field(default = None)

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
            doubleTimeCurrent = self.doubleTimeCurrent + other.doubleTimeCurrent,
            lengthGaugeCurrent = self.lengthGaugeCurrent + other.lengthGaugeCurrent
                if self.lengthGaugeCurrent is not None and other.lengthGaugeCurrent is not None
                else None
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
            doubleTimeCurrent = self.doubleTimeCurrent / other,
            lengthGaugeCurrent = self.lengthGaugeCurrent / other
                if self.lengthGaugeCurrent is not None
                else None
        )