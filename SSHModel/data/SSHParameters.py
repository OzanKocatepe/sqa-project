from dataclasses import dataclass, field
import numpy as np

@dataclass
class EnsembleParameters:
    """Contains the parameters for an SSH ensemble."""
    t1: float
    t2: float
    decayConstant: float
    drivingAmplitude: float
    drivingFreq: float

@dataclass
class ModelParameters(EnsembleParameters):
    """Contains the parameters for a particular SSH model."""
    k: float

    # Terms that will be defined from the momentum and other parameters.
    Ek: float = None
    phiK: float = None

    def __post_init__(self):
        """Defines useful values after the main parameters are initialised."""

        self.Ek = self.t1 + self.t2 * np.exp(1j * self.k)
        self.phiK = np.angle(self.Ek)

    @classmethod
    def FromEnsemble(cls, k: float, ensemble: EnsembleParameters) -> ModelParameters:
        """
        Creates an instance of the class from an instance of EnsembleParameters.
        
        Parameters
        ----------
        k : float
            The momentum of this particular SSH model.
        ensemble : EnsembleParameters
            The parameters of the ensemble to contain.

        Returns
        -------
            An instance of ModelParameters which contains the relevant parameters.
        """

        return cls(
            k = k,
            t1 = ensemble.t1,
            t2 = ensemble.t2,
            decayConstant = ensemble.decayConstant,
            drivingAmplitude = ensemble.drivingAmplitude,
            drivingFreq = ensemble.drivingFreq
        )