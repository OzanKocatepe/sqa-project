import numpy as np
import scipy

class InternalMixin:
    """
    Contains functions that encode similar logic, just with operators swapped out.
    
    Methods
    -------
    _RotateToBandBasis: Rotates an operator - (2, 2) matrix or stack of (2, 2) matrices - into the band basis.
    _GetMinus: Gets the coefficient of sigma_- for an operator in the band basis.
    _GetPlus: Gets the coefficient of sigma_+ for an operator in the band basis.
    _GetZ: Gets the coefficient of sigma_z for an operator in the band basis.
    """

    def _RotateToBandBasis(self,
        operator: np.ndarray[complex]
    ) -> np.ndarray[complex]:
        """
        Gets the operator in the band basis.

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

        # Rotates operator numerically using the calculated U matrix.
        # eigenmatrix = self.U.conj().T @ operator @ self.U

        eigenmatrix = np.zeros_like(operator, dtype=complex)

        # Calculate diagonal components.
        eigenmatrix[:, 0, 0] = np.trace(self.PPlus @ operator, axis1=1, axis2=2)
        eigenmatrix[:, 1, 1] = np.trace(self.PMinus @ operator, axis1=1, axis2=2)

        # Pick arbitrary vector r.
        r = 1 / np.sqrt(2) * (self.plusEigenvector + self.minusEigenvector)
        # Calculate off-diagonal components.
        denominator = np.sqrt(r.conj().T @ self.PPlus @ r @ r.conj().T @ self.PPlus @ r)
        eigenmatrix[:, 0, 1] = (r.conj().T @ self.PPlus @ operator @ self.PMinus @ r).squeeze() / denominator
        eigenmatrix[:, 1, 0] = (r.conj().T @ self.PMinus @ operator @ self.PPlus @ r).squeeze() / denominator

        # if np.sum(np.abs(eigenmatrix[:, 0, 1] - np.conjugate(eigenmatrix[:, 1, 0])) > 1e-1) > 0:
        #     print(np.abs(eigenmatrix[:, 0, 1] - np.conjugate(eigenmatrix[:, 1, 0]))[np.abs(eigenmatrix[:, 0, 1] - np.conjugate(eigenmatrix[:, 1, 0])) > 1e-1])
        #     raise ValueError("Off-diagonal components aren't conjugates.")
 
        # Squeezes eigenmatrix to deal with the case when the shape is
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

        bandOperator = self._RotateToBandBasis(operator)

        # Changes the indexing between a single operator and a stack of operators.
        if bandOperator.ndim == 2:
            return 0.5 * (bandOperator[1, 0] + np.conjugate(bandOperator[0, 1]))
        else:
            return 0.5 * (bandOperator[:, 1, 0] + np.conjugate(bandOperator[:, 0, 1]))

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

        bandOperator = self._RotateToBandBasis(operator)

        # Changes the indexing between a single operator and a stack of operators.
        if bandOperator.ndim == 2:
            return 0.5 * (bandOperator[0, 1] + np.conjugate(bandOperator[1, 0]))
        else:
            return 0.5 * (bandOperator[:, 0, 1] + np.conjugate(bandOperator[:, 1, 0]))

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

        bandOperator = self._RotateToBandBasis(operator)

        # Changes the indexing between a single operator and a stack of operators.
        if bandOperator.ndim == 2:
            return 0.5 * (bandOperator[0, 0] - bandOperator[1, 1])
        else:
            return 0.5 * (bandOperator[:, 0, 0] - bandOperator[:, 1, 1])