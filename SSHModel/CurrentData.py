from dataclasses import dataclass
import numpy as np

@dataclass
class CurrentData:
    """Contains all of the data relating to the current operator."""
    timeDomainData: np.ndarray[complex]
    freqDomainData: np.ndarray[complex]
    freqAxis: np.ndarray[float]