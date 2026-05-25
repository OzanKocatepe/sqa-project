import numpy as np
import jax.numpy as jnp
from functools import cache
from scipy import special

from data import ModelParameters
from data.band_basis import BandBasis

"""The Hamiltonian operator."""

sigmax = np.array([[0, 1],
                    [1, 0]], dtype=complex)

sigmay = np.array([[0, -1j],
                    [1j, 0]], dtype=complex)

sigmaz = np.array([[1, 0],
                    [0, -1]], dtype=complex)

def Ax(params: ModelParameters, t: float | jnp.ndarray[float]) -> float | jnp.ndarray[float]:
    """
    Returns the value of the driving field in the x-direction at time t.

    Parameters
    ----------
    params : ModelParameters
        The parameters of the model.
    t : float | ndarray[float]
        The time, in seconds, at which to evaluate the Hamiltonian.
        Accepts vectorised ijnputs.

    Returns
    -------
    float | ndarray[float]:
        The value of the driving field in the x-direction at time(s) t.
        The type returned is the same as the type of t.
    """

    return params.drivingAmp * jnp.sin(params.angularFreq * t)

def hx(params: ModelParameters, t: float | jnp.ndarray[float]=0) -> float | jnp.ndarray[float]:
    """
    Returns the coefficient of sigma_x in the driven Hamiltonian at time t.
    
    Parameters
    ----------
    params : ModelParameters
        The parameters of the model.
    t : float | ndarray[float], optional
        The time, in seconds, at which to evaluate the Hamiltonian.
        Accepts vectorised ijnputs.
        If called without a time value (or with t = 0), returns the coefficient of
        sigma_x in the undriven Hamiltonian.

    Returns
    -------
    float | ndarray[float]:
        The coefficient of sigma_x in the driven Hamiltonian at time(s) t.
        The type returned is the same as the type of t, if given.
        If no t is given, returns a float.
    """

    return jnp.sin(params.kx - Ax(params, t))

# @cache   
def hy(params: ModelParameters) -> float:
    """
    Returns the coefficient of sigma_y in the Hamiltonian.
    This result is cached, since we are only driving in the x-direction and hence
    this component has no dependence on time, and the momentum doesn't change once given.

    Parameters
    ----------
    params : ModelParameters
        The parameters of the model.
    
    Returns
    -------
    float:
        The coefficient of sigma_y in the Hamiltonian.
    """

    return jnp.sin(params.ky)

def hz(params: ModelParameters, t: float | jnp.ndarray[float]=0) -> float | jnp.ndarray[float]:
    """
    Returns the coefficient of sigma_z in the driven Hamiltonian at time t.
    
    Parameters
    ----------
    params : ModelParameters
        The parameters of the model.
    t : float | ndarray[float], optional
        The time, in seconds, at which to evaluate the Hamiltonian.
        Accepts vectorised ijnputs.
        If called without a time value (or with t = 0), returns the coefficient of
        sigma_z in the undriven Hamiltonian.

    Returns
    -------
    float | ndarray[float]:
        The coefficient of sigma_z in the driven Hamiltonian at time(s) t.
        The type returned is the same as the type of t, if given.
        If no t is given, returns a float.
    """

    return params.delta + jnp.cos(params.kx - Ax(params, t)) + jnp.cos(params.ky)

# @cache
def energy(params: ModelParameters) -> float:
    """
    Returns the ujnperturbed energy of the system at this momentum point.
    This result is cached, since the ujnperturbed energy has no dependence on time,
    and the momentum doesn't change once given.

    Returns
    -------
    float:
        The energy of the ujnperturbed system at this momentum.
    """

    return jnp.sqrt(hx(params)**2 + hy(params)**2 + hz(params)**2)

def lattice_basis(params: ModelParameters, t: float | jnp.ndarray[float]=0) -> jnp.ndarray[complex]:
    """Calculates the Hamiltonian operator in the lattice basis.
    
    Parameters
    ----------
    params : ModelParameters
        The parameters of the model
    t : float | ndarray[float], optional
        The times at which to evaluate the operator. Can be vectorised.
    """

    t = jnp.atleast_1d(t)

    H = jnp.multiply.outer(hx(params, t), sigmax) \
        + (hy(params) * sigmay)[jnp.newaxis, :, :] \
        + jnp.multiply.outer(hz(params, t), sigmaz)
    
    return H.squeeze()

def hxn(params: ModelParameters, n: int | np.ndarray[int]) -> complex | np.ndarray[complex]:
    """
    Calculates the nth Fourier coefficient for the driven term hx(t).

    Parameters
    ----------
    params : ModelParameters
        The parameters of the model
    n : int | ndarray[int]
        The index, or indices, that we will calculate the Fourier coefficients of.
        The index corresponds to the harmonic of the base frequency.

    Returns
    -------
    complex | ndarray[complex]:
        The desired coefficient(s). Of the same type as n.
    """

    # Form numpy array so that we can iterate through n.
    n = np.atleast_1d(n)
    coeffs = np.zeros_like(n, dtype=complex)

    zeroMask = n == 0
    evenMask = n % 2 == 0
    oddMask = ~evenMask

    # Calculates the relevant coefficients in a vectorised manner.
    # Not sure if scipy is actually faster vectorised, but regardless it neatens up the code.
    coeffs[oddMask] = np.sign(n[oddMask]) * 1j * special.jv(np.abs(n[oddMask]), params.drivingAmp) * np.cos(params.kx)
    coeffs[evenMask] = special.jv(np.abs(n[evenMask]), params.drivingAmp) * np.sin(params.kx)
    coeffs[zeroMask] = special.jv(0, params.drivingAmp) * np.sin(params.kx)

    # Returns the array as a float if it has size 1.
    if coeffs.size == 1:
        return coeffs[0]

    return coeffs

def hyn(params: ModelParameters, n: int | np.ndarray[int]) -> complex | np.ndarray[complex]:
    """
    Calculates the nth Fourier coefficient for the term hyn.
    This is a constant term, so returning the 'Fourier coefficients' just
    returns the value itself for n = 0, and 0 for all other n.
    This function is just a utility to use for compatibility with the other Fourier
    coefficient functions.

    Parameters
    ----------
    params : ModelParameters
        The parameters of the model
    n : int | ndarray[int]
        The index, or indices, that we will calculate the Fourier coefficients of.
        The index corresponds to the harmonic of the base frequency.

    Returns
    -------
    complex | ndarray[complex]:
        The desired coefficient(s). Of the same type as n.
    """

    # Form numpy array so that we can iterate through n.
    n = np.atleast_1d(n)
    coeffs = np.zeros_like(n, dtype=complex)

    # Sets n = 0 to the value of the constant,
    # otherwise leaves all other coefficients as zero.
    coeffs[n == 0] = hy(params)

    # Returns the array as a float if it has size 1.
    if coeffs.size == 1:
        return coeffs[0]

    return coeffs


def hzn(params: ModelParameters, n: int | np.ndarray[int]) -> complex | np.ndarray[complex]:
    """
    Calculates the nth Fourier coefficient for the driven term hz(t).

    Parameters
    ----------
    params : ModelParameters
        The parameters of the model
    n : int | ndarray[int]
        The index, or indices, that we will calculate the Fourier coefficients of.
        The index corresponds to the harmonic of the base frequency.

    Returns
    -------
    complex | ndarray[complex]:
        The desired coefficient(s). Of the same type as n.
    """

    # Form numpy array so that we can iterate through n.
    n = np.atleast_1d(n)
    coeffs = np.zeros_like(n, dtype=complex)

    zeroMask = n == 0
    evenMask = n % 2 == 0
    oddMask = ~evenMask

    # Calculates the relevant coefficients in a vectorised manner.
    # Not sure if scipy is actually faster vectorised, but regardless it neatens up the code.
    coeffs[evenMask] = special.jv(np.abs(n[evenMask]), params.drivingAmp) * np.cos(params.kx)
    coeffs[oddMask] = np.sign(n[oddMask]) * -1j * special.jv(np.abs(n[oddMask]), params.drivingAmp) * np.sin(params.kx)
    coeffs[zeroMask] = params.delta + np.cos(params.kx) * special.jv(0, params.drivingAmp) \
                + np.cos(params.ky)

    # Returns the array as a float if it has size 1.
    if coeffs.size == 1:
        return coeffs[0]

    return coeffs

def lattice_fourier_coefficient(params: ModelParameters, n: int | np.ndarray[int]) -> complex | np.ndarray[complex]:
    """Calculates the fourier series coefficients of the Hamiltonian in the lattice basis.
    
    Parameters
    ----------
    params : ModelParameters
        The parameters of the model.
    n : int | ndarray[int]
        The indices of the coefficients to calculate. Can be vectorised.

    Returns
    -------
    ndarray[complex]
        The corresponding fourier series coefficients.
    """

    n = np.atleast_1d(n)

    H = np.array([[hzn(params, n), hxn(params, n) - 1j * hyn(params, n)],
                    [hxn(params, n) + 1j * hyn(params, n), -hzn(params, n)]], dtype=complex)
    
    return np.moveaxis(H, -1, 0).squeeze()

def get_band_basis(params: ModelParameters) -> BandBasis:
    _, U = jnp.linalg.eigh(lattice_basis(params))
    U = jnp.flip(U, axis=1)

    return BandBasis(
        plusEigenvector = U[:, 0].reshape(2, 1),
        minusEigenvector = U[:, 1].reshape(2, 1),
        plusProjection = 0.5 * (np.eye(2) + lattice_basis(params) / energy(params)),
        minusProjection = 0.5 * (np.eye(2) - lattice_basis(params) / energy(params)),
    ) 