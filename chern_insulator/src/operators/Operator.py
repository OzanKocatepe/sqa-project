import numpy as np
from abc import ABC, abstractmethod

from data import ModelParameters
from operators import Hamiltonian
from operators.BandBasisProjector import BandBasisProjector

class Operator(ABC):
    """Abstract base class that all operators must inherit from.
    
    The base class insures that all operators have a representation in the lattice basis.
    The eigenbasis representation can be explicitly given, otherwise it is calculated numerically
    by rotating the lattice matrix into the band basis. Furthermore, the sigma_-, sigma_+, and sigma_z coefficients
    will be automatically calculated based on the band basis representation.

    If applicable, a fourier representation can also be defined.
    """

    def __init__(self, params: ModelParameters, hamiltonian: Hamiltonian):
        """Instantiates an instance of the operator.
        
        Parameters
        ----------
        params : ModelParameters
            The parameters of the model for which
            this operator will act on.
        hamiltonian : Hamiltonian
            The Hamiltonian of the system, which we will use to define the band basis.
        """

        self._params = params
        self._hamiltonian = hamiltonian

    @abstractmethod
    def lattice_basis(self, t: float | np.ndarray[float]) -> np.ndarray[complex]:
        """The representation of the operator in the lattice basis as a 2x2 matrix.
        
        Parameters
        ----------
        t : float | ndarray[float]
            The time, or times, at which to evaluate the operator.

        Returns
        -------
        ndarray[complex]
            The operator in the lattice basis at time(s) t. If t is a float, then
            returns an array of shape (2, 2). If t is an array of shape (n,), then returns an array
            of shape (n, 2, 2).
        """
        pass

    def band_basis(self, t: float | np.ndarray[float]) -> np.ndarray[complex]:
        """Rotates the operator into the band basis.

        Parameters
        ----------
        t : float | ndarray[float]
            The time, or times, at which to evaluate the operator.

        Returns
        -------
        ndarray[complex]
            Returns the operator in the band basis. If the input is shape (2, 2), returns
            a (2, 2) matrix. If the input is shape (n, 2, 2), returns an array of shape (n, 2, 2).
        """

        projector = BandBasisProjector(self._hamiltonian)
        return projector.rotate_to_band_basis(self.lattice_basis(t))

    def minus(self, t: float | np.ndarray[float]) -> complex | np.ndarray[complex]:
        """Gets the coefficient of sigma_- for this operator in the band basis.

        Parameters
        ----------
        t : float | ndarray[float]
            The time, or times, at which to evaluate the operator.

        Returns
        -------
        complex | ndarray[complex]
            The coefficient of sigma_- for this operator in the band basis.
            If t is a float, returns a scalar. If t is an array of shape (n,), returns an array of shape (n,).
        """

        bandOperator = self.band_basis(t)
        return BandBasisProjector.minus_coeff(bandOperator)

    def plus(self, t: float | np.ndarray[float]) -> complex | np.ndarray[complex]:
        """Gets the coefficient of sigma_+ for this operator in the band basis.

        Parameters
        ----------
        t : float | ndarray[float]
            The time, or times, at which to evaluate the operator.

        Returns
        -------
        complex | ndarray[complex]
            The coefficient of sigma_+ for this operator in the band basis.
            If t is a float, returns a scalar. If t is an array of shape (n,), returns an array of shape (n,).
        """

        bandOperator = self.band_basis(t)
        return BandBasisProjector.plus_coeff(bandOperator)

    def z(self, t: float | np.ndarray[float]) -> complex | np.ndarray[complex]:
        """Gets the coefficient of sigma_z for this operator in the band basis.

        Parameters
        ----------
        t : float | ndarray[float]
            The time, or times, at which to evaluate the operator.

        Returns
        -------
        complex | ndarray[complex]
            The coefficient of sigma_z for this operator in the band basis.
            If t is a float, returns a scalar. If t is an array of shape (n,), returns an array of shape (n,).
        """

        bandOperator = self.band_basis(t)
        return BandBasisProjector.z_coeff(bandOperator)