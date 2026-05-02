import numpy as np
from functools import cache, cached_property
from typing import override

from operators import Operator
from data import ModelParameters

class Hamiltonian(Operator):
    """The Hamiltonian operator."""

    def __init__(self, params: ModelParameters):
        """Overrides the base class' initialiser.
        
        The Hamiltonian defines the band basis, and so it doesn't make sense
        for it to take in a band basis and projection operators before it has
        been defined itself.

        Parameters
        ----------
        params : ModelParameters
            The parameters of the model for which
            the Hamiltonian will be calculated.
        """

        self._params = params

    def Ax(self, t: float | np.ndarray[float]) -> float | np.ndarray[float]:
        """
        Returns the value of the driving field in the x-direction at time t.

        Parameters
        ----------
        t : float | ndarray[float]
            The time, in seconds, at which to evaluate the Hamiltonian.
            Accepts vectorised inputs.

        Returns
        -------
        float | ndarray[float]:
            The value of the driving field in the x-direction at time(s) t.
            The type returned is the same as the type of t.
        """

        return self._params.drivingAmp * np.sin(self._params.angularFreq * t)

    def hx(self, t: float | np.ndarray[float]=0) -> float | np.ndarray[float]:
        """
        Returns the coefficient of sigma_x in the driven Hamiltonian at time t.
        
        Parameters
        ----------
        t : float | ndarray[float], optional
            The time, in seconds, at which to evaluate the Hamiltonian.
            Accepts vectorised inputs.
            If called without a time value (or with t = 0), returns the coefficient of
            sigma_x in the undriven Hamiltonian.

        Returns
        -------
        float | ndarray[float]:
            The coefficient of sigma_x in the driven Hamiltonian at time(s) t.
            The type returned is the same as the type of t, if given.
            If no t is given, returns a float.
        """

        return np.sin(self._params.kx - self.Ax(t))
 
    @cache   
    def hy(self) -> float:
        """
        Returns the coefficient of sigma_y in the Hamiltonian.
        This result is cached, since we are only driving in the x-direction and hence
        this component has no dependence on time, and the momentum doesn't change once given.
        
        Returns
        -------
        float:
            The coefficient of sigma_y in the Hamiltonian.
        """

        return np.sin(self._params.ky)
    
    def hz(self, t: float | np.ndarray[float]=0) -> float | np.ndarray[float]:
        """
        Returns the coefficient of sigma_z in the driven Hamiltonian at time t.
        
        Parameters
        ----------
        t : float | ndarray[float], optional
            The time, in seconds, at which to evaluate the Hamiltonian.
            Accepts vectorised inputs.
            If called without a time value (or with t = 0), returns the coefficient of
            sigma_z in the undriven Hamiltonian.

        Returns
        -------
        float | ndarray[float]:
            The coefficient of sigma_z in the driven Hamiltonian at time(s) t.
            The type returned is the same as the type of t, if given.
            If no t is given, returns a float.
        """

        return self._params.delta + np.cos(self._params.kx - self.Ax(t)) + np.cos(self._params.ky)

    @cache
    def energy(self) -> float:
        """
        Returns the unperturbed energy of the system at this momentum point.
        This result is cached, since the unperturbed energy has no dependence on time,
        and the momentum doesn't change once given.

        Returns
        -------
        float:
            The energy of the unperturbed system at this momentum.
        """

        return np.sqrt(self.hx()**2 + self.hy()**2 + self.hz()**2)

    def lattice_basis(self, t: float | np.ndarray[float]=0) -> np.ndarray[complex]:
        """
        Gets the value of the Hamiltonian in the lattice basis at time t.

        Parameters
        ----------
        t : float | ndarray[float]
            The time, in seconds, to evaluate the Hamiltonian at.
            If zero, this is the unperturbed Hamiltonian.

        Returns
        -------
        ndarray[complex]:
            The Hamiltonian evaluated at each time. Has the shape
            (t.size, 2, 2), or (2, 2) if t is a scalar.
        """

        t = np.atleast_1d(t)

        H = np.multiply.outer(self.hx(t), self.sigmax) \
            + (self.hy() * self.sigmay)[np.newaxis, :, :] \
            + np.multiply.outer(self.hz(t), self.sigmaz)
        
        return H.squeeze()
    
    @override
    def band_basis(self, t: float | np.ndarray[float]=0) -> np.ndarray[complex]:
        """Calculates the Hamiltonian in the band basis using an analytical formula.
        
        This overrides the normal function which rotates to the band basis because that first requires
        the eigenbasis of this Hamiltonian, which is clearly not defined yet. Furthermore, because of this
        exact reason we don't require the Hamiltonian to be given any band basis or projection operators
        because it creates them.
        
        Parameters
        ----------
        t : float | ndarray[float]
            The time, or times, at which to evaluate the Hamiltonian.
        
        Returns
        -------
        ndarray[complex]
            The Hamiltonian in the band basis (its unperturbed eigenbasis). If t = 0, the operator is guaranteed to be
            diagonal. Returns either a (n, 2, 2) array if t is a vector, or a (2, 2) array if t is a scalar.
        """

        return np.multiply.outer(self.energy(t), self.sigmaz)
    
    @cached_property
    def plusEigenvector(self) -> np.ndarray[complex]:
        _, U = np.linalg.eigh(self.lattice_basis())
        return U[:, 1].reshape(2, 1)
    
    @cached_property
    def minusEigenvector(self) -> np.ndarray[complex]:
        _, U = np.linalg.eigh(self.lattice_basis())
        return U[:, 0].reshape(2, 1)
    
    @cached_property
    def U(self) -> np.ndarray[complex]:
        U = np.column_stack((self.plusEigenvector, self.minusEigenvector))

        # Raise an exception if U is not unitary.
        if not np.allclose(U @ U.conj().T, np.eye(2)) or not np.allclose(U.conj().T @ U , np.eye(2)):
            raise ValueError("Matrix U is not unitary.")
        
        return U
    
    @cached_property
    def plusProjection(self) -> np.ndarray[complex]:
        return 0.5 * (np.eye(2) + self.lattice_basis() / self.energy())
    
    @cached_property
    def minusProjection(self) -> np.ndarray[complex]:
        return 0.5 * (np.eye(2) - self.lattice_basis() / self.energy())