import jax.numpy as np

from data.band_basis import BandBasis

def rotate_to_band_basis(basis: BandBasis, operator: np.ndarray[complex]) -> np.ndarray[complex]:
    """Rotates a matrix into the band basis.
    
    Parameters
    ----------
    basis : BandBasis
        The basis to rotate operator into.
    operator : ndarray[complex]
        The matrix to rotate into the band basis. Must have shape (2, 2) or
        (n, 2, 2) if it is a stack of matrices to rotate.

    Returns
    -------
    ndarray[complex]
        The matrix in the band basis. If the input is shape (2, 2), returns
        a (2, 2) matrix. If the input is shape (n, 2, 2), returns an array of shape (n, 2, 2).
    """

    # If we have a single operator of shape (2, 2), transforms it into
    # shape (1, 2, 2).
    if operator.ndim == 2:
        operator = operator[np.newaxis, :, :]

    # Rotates operator numerically using the calculated U matrix.
    # eigenmatrix = self.U.conj().T @ operator @ self.U

    # Rotates the operator numerically using the projection operators.
    eigenmatrix = np.zeros_like(operator, dtype=complex)

    # Calculate diagonal components.
    eigenmatrix = eigenmatrix.at[:, 0, 0].set(np.trace(basis.plusProjection @ operator, axis1=1, axis2=2))
    eigenmatrix = eigenmatrix.at[:, 1, 1].set(np.trace(basis.minusProjection @ operator, axis1=1, axis2=2))

    # Pick arbitrary vector r.
    r = 1 / np.sqrt(2) * (basis.plusEigenvector + basis.minusEigenvector)
    # Calculate off-diagonal components.
    denominator = np.sqrt(r.conj().T @ basis.plusProjection @ r @ r.conj().T @ basis.plusProjection @ r)
    eigenmatrix = eigenmatrix.at[:, 0, 1].set((r.conj().T @ basis.plusProjection @ operator @ basis.minusProjection @ r / denominator).squeeze())
    eigenmatrix = eigenmatrix.at[:, 1, 0].set((r.conj().T @ basis.minusProjection @ operator @ basis.plusProjection @ r / denominator).squeeze())

    # Squeezes eigenmatrix to deal with the case when the shape is
    # (1, 2, 2), so we will get the matrix back as a (2, 2) matrix.
    # Otherwise, returns an (n, 2, 2) matrix.
    # Indexing can happen outside the function since we will know what the return shape will be
    # based on our input.
    return eigenmatrix.squeeze()

def minus_coeff(operator: np.ndarray[complex]) -> complex | np.ndarray[complex]:
    """Gets the coefficient of sigma_- for this matrix.

    Parameters
    ----------
    operator: ndarray[complex]
        The matrix for which to calculate the coefficient
        of sigma_-. Must have shape (2, 2) or (n, 2, 2) if it is a stack of matrices to decompose.

    Returns
    -------
    complex | ndarray[complex]
        The coefficient of sigma_- for this matrix.
        If the input is shape (2, 2), returns a scalar. If the input is shape (n, 2, 2), returns
        an array of shape (n,).
    """

    # Changes the indexing between a single operator and a stack of operators.
    if operator.ndim == 2:
        return operator[1, 0]
    else:
        return operator[:, 1, 0]

def plus_coeff(operator: np.ndarray[complex]) -> complex | np.ndarray[complex]:
    """Gets the coefficient of sigma_+ for this matrix.

    Parameters
    ----------
    operator: ndarray[complex]
        The matrix for which to calculate the coefficient
        of sigma_+. Must have shape (2, 2) or (n, 2, 2) if it is a stack of matrices to decompose.

    Returns
    -------
    complex | ndarray[complex]
        The coefficient of sigma_+ for this matrix.
        If the input is shape (2, 2), returns a scalar. If the input is shape (n, 2, 2), returns
        an array of shape (n,).
    """

    # Changes the indexing between a single operator and a stack of operators.
    if operator.ndim == 2:
        return operator[0, 1]
    else:
        return operator[:, 0, 1]

def z_coeff(operator: np.ndarray[complex]) -> complex | np.ndarray[complex]:
    """Gets the coefficient of sigma_z for this matrix.

    Parameters
    ----------
    operator: ndarray[complex]
        The matrix for which to calculate the coefficient
        of sigma_z. Must have shape (2, 2) or (n, 2, 2) if it is a stack of matrices to decompose.

    Returns
    -------
    complex | ndarray[complex]
        The coefficient of sigma_z for this matrix.
        If the input is shape (2, 2), returns a scalar. If the input is shape (n, 2, 2), returns
        an array of shape (n,).
    """

    # Changes the indexing between a single operator and a stack of operators.
    if operator.ndim == 2:
        return 0.5 * (operator[0, 0] - operator[1, 1])
    else:
        return 0.5 * (operator[:, 0, 0] - operator[:, 1, 1])

def rotated_minus_coeff(basis: BandBasis, operator: np.ndarray[complex]) -> complex | np.ndarray[complex]:
    """Gets the coefficient of sigma_- for this matrix in the band basis.

    Parameters
    ----------
    operator: ndarray[complex]
        The matrix, in the lattice basis, for which to calculate the coefficient
        of sigma_-. Must have shape (2, 2) or (n, 2, 2) if it is a stack of matrices to decompose.

    Returns
    -------
    complex | ndarray[complex]
        The coefficient of sigma_- for this matrix in the band basis.
        If the input is shape (2, 2), returns a scalar. If the input is shape (n, 2, 2), returns
        an array of shape (n,).
    """

    return minus_coeff(rotate_to_band_basis(basis, operator))

def rotated_plus_coeff(basis: BandBasis, operator: np.ndarray[complex]) -> complex | np.ndarray[complex]:
    """Gets the coefficient of sigma_+ for this matrix in the band basis.

    Parameters
    ----------
    operator: ndarray[complex]
        The matrix, in the lattice basis, for which to calculate the coefficient
        of sigma_+. Must have shape (2, 2) or (n, 2, 2) if it is a stack of matrices to decompose.

    Returns
    -------
    complex | ndarray[complex]
        The coefficient of sigma_+ for this matrix in the band basis.
        If the input is shape (2, 2), returns a scalar. If the input is shape (n, 2, 2), returns
        an array of shape (n,).
    """

    return z_coeff(rotate_to_band_basis(basis, operator))

def rotated_z_coeff(basis: BandBasis, operator: np.ndarray[complex]) -> complex | np.ndarray[complex]:
    """Gets the coefficient of sigma_z for this matrix in the band basis.

    Parameters
    ----------
    operator: ndarray[complex]
        The matrix, in the lattice basis, for which to calculate the coefficient
        of sigma_z. Must have shape (2, 2) or (n, 2, 2) if it is a stack of matrices to decompose.

    Returns
    -------
    complex | ndarray[complex]
        The coefficient of sigma_z for this matrix in the band basis.
        If the input is shape (2, 2), returns a scalar. If the input is shape (n, 2, 2), returns
        an array of shape (n,).
    """

    return z_coeff(rotate_to_band_basis(basis, operator))