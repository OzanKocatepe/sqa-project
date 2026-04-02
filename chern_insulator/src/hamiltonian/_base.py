import numpy as np
from functools import cache, cached_property

from data import ModelParameters

class Base:
    """Contains the base logic for the Hamiltonian.

    Attributes
    ----------
    sigmax: The Pauli matrix sigma_x in the Schrodinger picture.
    sigmay: The Pauli matrix sigma_y in the Schrodinger picture.
    sigmaz: The Pauli matrix sigma_z in the Schrodinger picture.

    Cached Properties
    -----------------
    plusEigenvector: The eigenvector of the unperturbed Hamiltonian corresponding to the positive energy eigenvalue.
    minusEigenvector: The eigenvector of the unperturbed Hamiltonian corresponding to the negative energy eigenvalue.
    U: The unitary matrix constructed from these eigenvectors (first column is +, second column is -).
    
    Methods
    -------
    __init__: Initialises the Hamiltonian with the model parameters.
    Ax: Returns the value of the driving field in the x-direction at time t.
    H: Gets the value of the Hamiltonian in the lattice basis at time t.
    hx: Returns the coefficient of sigma_x in the driven Hamiltonian at time t.
    hy: Returns the coefficient of sigma_y in the Hamiltonian.
    hz: Returns the coefficient of sigma_z in the driven Hamiltonian at time t.
    energy: Returns the unperturbed energy of the system at this momentum point.
    StaticEnergy: Calculates the unperturbed energy of the system at a variety of momentums, vectorised.
    """

    sigmax = np.array([[0, 1],
                       [1, 0]], dtype=complex)
    
    sigmay = np.array([[0, -1j],
                       [1j, 0]], dtype=complex)

    sigmaz = np.array([[1, 0],
                       [0, -1]], dtype=complex)

    def __init__(self, params: ModelParameters) -> None:
        """
        Initialises the Hamiltonian with the model parameters.
        
        Parameters
        ----------
        params : ModelParameters
            The parameters of the Chern insulator model at this momentum.
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

    def H(self, t: float | np.ndarray[float]=0) -> np.ndarray[complex]:
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

        H = self.hx(t)[:, np.newaxis, np.newaxis] * self.sigmax + (self.hy() * self.sigmay)[np.newaxis, :, :] + self.hz(t)[: np.newaxis, np.newaxis] * self.sigmaz
        
        if t.size == 1:
            return H[0, :, :]
        return H

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

    @staticmethod
    def StaticEnergy(kx: float | np.ndarray[float], ky: float | np.ndarray[float], delta: float) -> float | np.ndarray[float]:
        """
        Exact same thing as energy, but it turns out to be useful to have a
        static instance of this function so that we can calculate
        energies throughout the Brillouin zone without relying on other properties.

        We also vectorise this function in the momentums to speed things up.

        Parameters
        ----------
        kx : float | ndarray[float]
            The x-component of the momentum.
        ky : float | ndarray[float]
            The x-component of the momentum.
            Must be the same type as kx.
        delta : float
            The value of delta.

        Returns
        -------
        float | ndarray[float]:
            The energy of the unperturbed system at this momentum
            and delta.
            Will be the same type as kx and ky.
        """

        hx = np.sin(kx)
        hy = np.sin(ky)
        hz = delta + np.cos(kx) + np.cos(ky)

        return np.sqrt(hx**2 + hy**2 + hz**2)  
    
    @cached_property
    def plusEigenvector(self) -> np.ndarray[complex]:
        _, U = np.linalg.eigh(self.H())
        return U[:, 1].reshape(2, 1)
    
    @cached_property
    def minusEigenvector(self) -> np.ndarray[complex]:
        _, U = np.linalg.eigh(self.H())
        return U[:, 0].reshape(2, 1)
    
    @cached_property
    def U(self) -> np.ndarray[complex]:
        return np.column_stack((self.plusEigenvector, self.minusEigenvector))