import numpy as np
from dataclasses import dataclass, field

@dataclass(slots=True)
class EnsembleParameters:
    """
    Parameters shared by the entire ensemble of Chern insulators at
    a range of momentum points.

    Attributes
    ----------
    delta : float
        The mass term in the Chern insulator Hamiltonian.
        Controls the topological phase of the system.
    drivingAmplitude : float
        The amplitude of the driving field in the x-direction.
    drivingFrequency : float
        The frequency of the driving field in the x-direction, in Hz.
    angularFreq : float
        The angular frequency of the driving field, in radians per second.
    decayConstant : float
        The decay constant for the system, in s^-1.
    """

    delta: float
    drivingAmplitude: float
    drivingFrequency: float
    decayConstant: float
    angularFreq: float = field(init=False)

    def __post_init__(self) -> None:
        """
        Calculates useful terms one time, rather than having to recalculate in
        the main code.
        """

        self.angularFreq = 2 * np.pi * self.drivingFrequency

@dataclass(slots=True)
class ModelParameters(EnsembleParameters):
    """
    Parameters specific to a single Chern insulator model at a single
    momentum point.
    """

    kx: float
    ky: float