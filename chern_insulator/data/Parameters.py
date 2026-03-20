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
    decayConstant : float
        The decay constant for the system, in s^-1.
    maxN : int
        The maximum harmonic that we will calculate Fourier series up to. For
        maxN = n, we will calculate the coefficients c_{-n} to c_n.
    angularFreq : float
        The angular frequency of the driving field, in radians per second.
    """

    delta: float
    drivingAmp: float
    drivingFreq: float
    decayConstant: float
    maxN: int

    angularFreq: float = field(init=False)

    def __post_init__(self) -> None:
        """
        Calculates useful terms one time, rather than having to recalculate in
        the main code.
        """

        self.angularFreq = 2 * np.pi * self.drivingFreq

@dataclass(slots=True)
class ModelParameters(EnsembleParameters):
    """
    Parameters specific to a single Chern insulator model at a single
    momentum point.
    """

    kx: float
    ky: float

    @staticmethod
    def FromEnsemble(kx: float, ky: float, params: EnsembleParameters) -> ModelParameters:
        """
        Creates an instance of ModelParameters from an instance of EnsembleParameters.

        Parameters
        ----------
        kx : float
            The x-component of the momentum.
        ky : float
            The y-component of the momentum.
        params : EnsembleParameters
            The object containing the remaining parameters required
            for the ModelParameters instance.

        Returns
        -------
        ModelParameters:
            An instance containing the same attributes as params, along with the
            given kx and ky.
        """
        
        return ModelParameters(
            kx = kx,
            ky = ky,
            delta = params.delta,
            drivingAmp = params.drivingAmp,
            drivingFreq = params.drivingFreq,
            decayConstant = params.decayConstant,
            maxN = params.maxN
        )