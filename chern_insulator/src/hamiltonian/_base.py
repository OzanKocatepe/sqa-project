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
    U: The unitary matrix constructed from these eigenvectors (first column is |+>, second column is |->).
    
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
 