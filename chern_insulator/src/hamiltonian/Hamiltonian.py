import numpy as np
from functools import cache, cached_property

from hamiltonian._base import Base
from hamiltonian._eigenbasis import EigenbasisMixin
from hamiltonian._current import CurrentMixin
from hamiltonian._fourier import FourierMixin
from hamiltonian._internal import InternalMixin
from hamiltonian._dynamics import DynamicsMixin
from hamiltonian._topology import TopologyMixin

class Hamiltonian(Base,
                  EigenbasisMixin,
                  CurrentMixin,
                  FourierMixin,
                  InternalMixin,
                  DynamicsMixin,
                  TopologyMixin):
    """Contains the functions derived from the Chern Model Hamiltonian."""
 
    @cached_property
    def rho(self) -> float:
        """
        Returns the value of rho, defined as sqrt(hx^2 + hy^2) where hx and hy are undriven terms.
        This is a commonly used term in the calculations.
        This result is cached, since rho has no time dependence.

        Returns
        -------
        float:
            The value of rho at this momentum point.
        """

        raise DeprecationWarning()

        return np.sqrt(self.hx()**2 + self.hy()**2)