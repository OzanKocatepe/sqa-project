import numpy as np

from . import hamiltonian
from data import ModelParameters

class ParamagneticCurrentX:
    """The paramagnetic current operator in the x-direction."""

    @staticmethod
    def lattice_basis(params: ModelParameters, t: float | np.ndarray[float]) -> np.ndarray[complex]: 
        t = np.atleast_1d(t)

        jpx = np.multiply.outer(-np.cos(params.kx - hamiltonian.Ax(params, t)), hamiltonian.sigmax) \
            + np.multiply.outer(np.sin(params.kx - hamiltonian.Ax(params, t)), hamiltonian.sigmaz)
         
        return jpx.squeeze()
    
class ParamagneticCurrentY:
    """The paramagnetic current operator in the y-direction."""

    @staticmethod
    def lattice_basis(params: ModelParameters ,t: float | np.ndarray[float]=0) -> np.ndarray[complex]:
        jpy = -np.cos(params.ky) * hamiltonian.sigmay + np.sin(params.ky) * hamiltonian.sigmaz
        
        return np.stack((jpy,) * t.size, axis = 0)
    
class DiamagneticCurrentXX:
    """The diamagnetic current operator in the x-direction."""

    @staticmethod
    def lattice_basis(params: ModelParameters, t: float | np.ndarray[float]) -> np.ndarray[complex]: 
        t = np.atleast_1d(t)
    
        jdx = (
            np.multiply.outer(-np.sin(params.kx - hamiltonian.Ax(t)), hamiltonian.sigmax)
            + np.multiply.outer(-np.cos(params.kx - hamiltonian.Ax(t)), hamiltonian.sigmaz)
        )

        return jdx.squeeze()