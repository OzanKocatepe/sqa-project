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

    paramagneticCurrent: np.ndarray[complex] = field(init = False)
    diamagneticCurrent: np.ndarray[complex] = field(init = False)

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
            diamagneticCurrent = self.diamagneticCurrent + other.diamagneticCurrent
        )