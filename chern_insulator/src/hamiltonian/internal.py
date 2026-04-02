from typing import Callable
import inspect
import numpy as np

class InternalMixin:
    """Contains functions that encode similar logic, just with operators swapped out."""

    def _RotateToBandBasis(self,
        operator: np.ndarray[complex]
    ) -> np.ndarray[complex]:
        """
        Gets the operator in its eigenbasis.

        Parameters
        ----------
        operator : ndarray[complex]
            The operator to find the band basis representation of.
            Should be a (2, 2) matrix, or a stack of matrices of the shape
            (n, 2, 2).

        Returns
        -------
        ndarray[complex]
            Returns the operator in the band basis. If the input is shape (2, 2), returns
            a (2, 2) matrix. If the input is shape (n, 2, 2), returns an array of shape (n, 2, 2).
        """

        # If we have a single operator of shape (2, 2), transforms it into
        # shape (1, 2, 2).
        if operator.ndim == 2:
            operator = operator[np.newaxis, :, :]

        eigenmatrix = (self.U.conj().T @ operator @ self.U)
        
        # Squeezes eigenmatrix to deal with the case whne the shape is
        # (1, 2, 2), so we will get the matrix back as a (2, 2) matrix.
        # Otherwise, returns an (n, n) matrix.
        # Indexing can happen outside the function since we will know what the return shape will be
        # based on our input.
        return eigenmatrix.squeeze()

    def _GetMinus(self,
        operator: np.ndarray[complex]
    ) -> complex | np.ndarray[complex]:
        """
        Gets the coefficient of sigma_- for this operator in the band basis.

        Parameters
        ----------
        operator : ndarray[complex]
            The operator to find the coefficient for.
            Should be a (2, 2) matrix, or a stack of matrices of the shape
            (n, 2, 2).

        Returns
        -------
        complex | ndarray[complex]
            The coefficient of sigma_- for this operator in the band basis.
            If the input is shape (2, 2), returns a scalar. If the input is shape (n, 2, 2), returns an array of shape (n,).
        """

        return self._RotateToBandBasis(operator)[1, 0]

    def _GetPlus(self,
        operator: np.ndarray[complex]
    ) -> complex | np.ndarray[complex]:
        """
        Gets the coefficient of sigma_+ for this operator in the band basis.

        Parameters
        ----------
        operator : ndarray[complex]
            The operator to find the coefficient for.
            Should be a (2, 2) matrix, or a stack of matrices of the shape
            (n, 2, 2).

        Returns
        -------
        complex | ndarray[complex]
            The coefficient of sigma_+ for this operator in the band basis.
            If the input is shape (2, 2), returns a scalar. If the input is shape (n, 2, 2), returns an array of shape (n,).
        """

        return self._RotateToBandBasis(operator)[0, 1]

    def _GetZ(self,
        operator: np.ndarray[complex]
    ) -> complex | np.ndarray[complex]:
        """
        Gets the coefficient of sigma_z for this operator in the band basis.

        Parameters
        ----------
        operator : ndarray[complex]
            The operator to find the coefficient for.
            Should be a (2, 2) matrix, or a stack of matrices of the shape
            (n, 2, 2).

        Returns
        -------
        complex | ndarray[complex]
            The coefficient of sigma_z for this operator in the band basis.
            If the input is shape (2, 2), returns a scalar. If the input is shape (n, 2, 2), returns an array of shape (n,).
        """

        return 0.5 * (self._RotateToBandBasis(operator)[0, 0] + self._RotateToBandBasis(operator)[1, 1])