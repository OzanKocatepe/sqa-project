import numpy as np

from .Operator import Operator

class ParamagneticCurrentX(Operator):
    """The paramagnetic current operator in the x-direction."""

    def lattice_basis(self, t: float | np.ndarray[float]) -> np.ndarray[complex]: 
        t = np.atleast_1d(t)

        jpx = np.multiply.outer(-np.cos(self._params.kx - self._hamiltonian.Ax(t)), self._hamiltonian.sigmax) \
            + np.multiply.outer(np.sin(self._params.kx - self._hamiltonian.Ax(t)), self._hamiltonian.sigmaz)
         
        return jpx.squeeze()
    
class ParamagneticCurrentY(Operator):
    """The paramagnetic current operator in the y-direction."""

    def lattice_basis(self, t: float | np.ndarray[float]=0) -> np.ndarray[complex]:
        jpy = -np.cos(self._params.ky) * self._hamiltonian.sigmay + np.sin(self._params.ky) * self._hamiltonian.sigmaz
        
        return np.stack((jpy,) * t.size, axis = 0)
    
class DiamagneticCurrentX(Operator):
    """The diamagnetic current operator in the x-direction."""

    def lattice_basis(self, t: float | np.ndarray[float]) -> np.ndarray[complex]: 
        t = np.atleast_1d(t)
    
        jdx = (
            np.multiply.outer(-np.sin(self._params.kx - self._hamiltonian.Ax(t)), self._hamiltonian.sigmax)
            + np.multiply.outer(-np.cos(self._params.kx - self._hamiltonian.Ax(t)), self._hamiltonian.sigmaz)
        )

        return jdx.squeeze()