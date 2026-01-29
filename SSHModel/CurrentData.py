from dataclasses import dataclass
import numpy as np

@dataclass
class CurrentData:
    """Contains all of the data relating to the current operator."""
    timeDomainData: np.ndarray[complex]
    freqDomainData: np.ndarray[complex]
    freqAxis: np.ndarray[float]

    def __init__(self, timeDomainData: np.ndarray[complex], freqDomainData: np.ndarray[complex], freqAxis: np.ndarray[float]):
        """
        Constructs an instance of CurrentData. Can be instantiated with only some of the values upon creation.

        Parameters
        ----------
        timeDomainData : ndarray[complex]
            The values of the expectation of the current operator in the time-domain.
        freqDomainData: ndarray[complex]
            The same as timeDomainData, but in the frequency domain.
        freqAxis : ndarray[float]
            The frequencies that correspond to the values in freqDomainData
        """

        self.timeDomainData = timeDomainData
        self.freqDomainData = freqDomainData
        self.freqAxis = freqAxis