from dataclasses import dataclass, field
import numpy as np

@dataclass
class SSHParameters:
    """Contains the parameters for an SSH system."""
    t1: float
    t2: float
    decayConstant: float
    drivingAmplitude: float
    drivingFreq: float

    Ek: float = field(init = False)
    phiK: float = field(init = False)

    def CalculateUsefulTerms(self, k: float):
        """Defines useful values after the main parameters are defined."""
        self.Ek = self.t1 + self.t2 * np.exp(1j * k)
        self.phiK = np.angle(self.Ek)