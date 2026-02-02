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

ensemble = EnsembleParameters(
    t1 = 1,
    t2 = 2,
    decayConstant = 0.1,
    drivingAmplitude = 0.2,
    drivingFreq = 2 / 3.01
)

print(ensemble)

model = ModelParameters.FromEnsemble(
    k = np.pi / 4,
    ensemble = ensemble
)

print(model)