from dataclasses import dataclass, field

from .Fourier import Fourier

@dataclass(slots=True)
class CorrelationData:
    """
    Stores the correlation data for a model.

    Parameters
    ----------
    singleTimeFourier : list[Fourier]
        A list containing the Fourier series for sigma_-, sigma_+, and sigma_z.
    """

    singleTimeFourier: list[Fourier] = field(init = False)