import numpy as np
from abc import ABC, abstractmethod

from data import ModelParameters

class Operator(ABC):
    """Abstract base class that all operators must inherit from.
    
    The base class insures that all operators have a representation in the lattice basis.
    The eigenbasis representation can be explicitly given, otherwise it is calculated numerically
    by rotating the lattice matrix into the band basis. Furthermore, the sigma_-, sigma_+, and sigma_z coefficients
    will be automatically calculated based on the band basis representation.

    If applicable, a fourier representation can also be defined.
    """

    sigmax = np.array([[0, 1],
                       [1, 0]], dtype=complex)
    
    sigmay = np.array([[0, -1j],
                       [1j, 0]], dtype=complex)

    sigmaz = np.array([[1, 0],
                       [0, -1]], dtype=complex)

    def __init__(self, params: ModelParameters,
                       bandBasis: np.ndarray[complex],
                       plusProjection: np.ndarray[complex],
                       minusProjection: np.ndarray[complex]
                ):
        """Instantiates an instance of the operator.
        
        Parameters
        ----------
        params : ModelParameters
            The parameters of the model for which
            this operator will act on.
        bandBasis : ndarray[complex]
            A (2, 2) matrix such that bandBasis[:, 0] is the positive eigenvector and
            bandBasis[:, 1] is the negative eigenvector of the band basis.
        plusProjection : ndarray[complex]
            The shape (2, 2) projection operator into the positive eigenvector
            of the band basis.
        minusProjection : ndarray[complex]
            The shape (2, 2) projection operator onto the negative eigenvector
            of the band basis.
        """

        self._params = params
        self._bandBasis = bandBasis
        self._plusProjection = plusProjection
        self._minusProjection = minusProjection

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

        operator = self.lattice_basis(t)

        # If we have a single operator of shape (2, 2), transforms it into
        # shape (1, 2, 2).
        if operator.ndim == 2:
            operator = operator[np.newaxis, :, :]

        # Rotates operator numerically using the calculated U matrix.
        # eigenmatrix = self.U.conj().T @ operator @ self.U

        # Rotates the operator numerically using the projection operators.
        eigenmatrix = np.zeros_like(operator, dtype=complex)

        # Calculate diagonal components.
        eigenmatrix[:, 0, 0] = np.trace(self._plusProjection @ operator, axis1=1, axis2=2)
        eigenmatrix[:, 1, 1] = np.trace(self._minusProjection @ operator, axis1=1, axis2=2)

        plusEigenvector = self.bandBasis[:, 0]
        minusEigenvector = self.bandBasis[:, 1]

        # Pick arbitrary vector r.
        r = 1 / np.sqrt(2) * (plusEigenvector + minusEigenvector)
        # Calculate off-diagonal components.
        denominator = np.sqrt(r.conj().T @ self._plusProjection @ r @ r.conj().T @ self._plusProjection @ r)
        eigenmatrix[:, 0, 1] = (r.conj().T @ self._plusProjection @ operator @ self._minusProjection @ r).squeeze() / denominator
        eigenmatrix[:, 1, 0] = (r.conj().T @ self._minusProjection @ operator @ self._plusProjection @ r).squeeze() / denominator
 
        # Squeezes eigenmatrix to deal with the case when the shape is
        # (1, 2, 2), so we will get the matrix back as a (2, 2) matrix.
        # Otherwise, returns an (n, n) matrix.
        # Indexing can happen outside the function since we will know what the return shape will be
        # based on our input.
        return eigenmatrix.squeeze()

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

        # Changes the indexing between a single operator and a stack of operators.
        if bandOperator.ndim == 2:
            return bandOperator[1, 0]
        else:
            return bandOperator[:, 1, 0]

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

        # Changes the indexing between a single operator and a stack of operators.
        if bandOperator.ndim == 2:
            return bandOperator[0, 1]
        else:
            return bandOperator[:, 0, 1]

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

        # Changes the indexing between a single operator and a stack of operators.
        if bandOperator.ndim == 2:
            return 0.5 * (bandOperator[0, 0] - bandOperator[1, 1])
        else:
            return 0.5 * (bandOperator[:, 0, 0] - bandOperator[:, 1, 1])