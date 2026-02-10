from dataclasses import dataclass
import numpy as np
from .Fourier import Fourier

@dataclass
class CorrelationData:
    """Contains all of the data relating to the single- and double-time correlations."""

    singleTime: np.ndarray[complex] = None
    doubleTime: np.ndarray[complex] = None
    singleFourierSeries: list[Fourier] = None

    def __add__(self, other: CorrelationData) -> CorrelationData:
        """
        Adds another CorrelationData instance to itself to create a new CorrelationData instance.

        Parameters
        ----------
        other : CorrelationData
            The other CorrelationData object to add to this one.

        Returns
        -------
        CorrelationData
            The new current data instance, the sum of their time and fourier data.
        """

        return CorrelationData(
            singleTime = self.singleTime + other.singleTime,
            doubleTime = self.doubleTime + other.doubleTime,
            singleFourierSeries = self.singleFourierSeries + other.singleFourierSeries
        )