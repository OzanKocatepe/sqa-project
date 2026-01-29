from dataclasses import dataclass

@dataclass
class SSHParameters:
    """Contains the parameters for an SSH system."""
    t1: float
    t2: float
    decayConstant: float
    drivingAmplitude: float
    drivingFreq: float