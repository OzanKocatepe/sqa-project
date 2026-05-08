from typing import Protocol
import numpy as np

class BandBasisProtocol(Protocol):
    """Defines the protocol that a class must follow to define a band basis.
    
    To rotate our operators into the band basis, we need the Hamiltonian of the system in order to define
    what the band basis is. However, we don't need the entire Hamiltonian instance - we only need the properties
    relating to the eigenbasis of the unperturbed Hamiltonian.
    
    Hence, we can define the protocol which the Hamiltonian class will follow in order to
    define the band basis. This way, Hamiltonian can import from Operator, and Operator can have a
    BandBasisProtocol instance, without requiring a full Hamiltonian instance (which would require
    a circular import).
    """

    plusEigenvector: np.ndarray[complex]
    minusEigenvector: np.ndarray[complex]
    plusProjection: np.ndarray[complex]
    minusProjection: np.ndarray[complex]