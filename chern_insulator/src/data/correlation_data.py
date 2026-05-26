from dataclasses import dataclass, field
import numpy as np

from .fourier import Fourier

@dataclass(slots=True)
class CorrelationData:
    """
    Stores the correlation data for a model.

    Parameters
    ----------
    singleTimeFourier : list[Fourier]
        A list containing the Fourier series for sigma_-, sigma_+, and sigma_z.
    doubleTimeCorrelations: ndarray[complex]
        A (3, 3, time.size) array containing the double-time correlations.
        The first axis represents the left operator, the second axis represents the right operator.
        For both axes, the indices 0, 1, and 2 correspond to sigma_-, sigma_+, sigma_z respectively.
    """

    first_order_fourier: list[Fourier] = field(init = False)
    second_order_correlations: np.ndarray[complex] = field(init = False)