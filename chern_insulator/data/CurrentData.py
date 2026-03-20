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