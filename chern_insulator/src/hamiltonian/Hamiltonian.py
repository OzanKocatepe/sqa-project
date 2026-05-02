from hamiltonian._base import Base
from hamiltonian._eigenbasis import EigenbasisMixin
from hamiltonian._current import CurrentMixin
from hamiltonian._fourier import FourierMixin
from hamiltonian._dynamics import DynamicsMixin
from hamiltonian._topology import TopologyMixin
from hamiltonian._lengthgauge import LengthGaugeMixin

class Hamiltonian(Base,
                  EigenbasisMixin,
                  CurrentMixin,
                  FourierMixin,
                  DynamicsMixin,
                  TopologyMixin,
                  LengthGaugeMixin):
    """Contains the functions derived from the Chern Model Hamiltonian.""" 