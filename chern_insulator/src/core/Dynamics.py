import numpy as np
import jax
import jax.numpy as jnp

from data import ModelParameters, DoubleTimeODEParams
from operators import hamiltonian

"""Stores all information about dynamics of the system - specifically the
equations of motion for single- and double-time correlations.
"""

# @jax.jit
def _lattice_basis_hamiltonian(t: float, odeParams: DoubleTimeODEParams) -> jnp.ndarray[jnp.complex64]:
    """Calculates the lattice basis Hamiltonian at time t.
        
    Duplicate of a function found in the hamiltonian class, but that is necessary to be able
    utilise jax.

    parameters
    ----------
    t : float
        the time, in seconds, to evaluate the hamiltonian at.
    odeparams : doubletimeodeparams
        the relevant parameters for calculation of the hamiltonian.

    Returns
    -------
    ndarray[complex]
        The lattice basis Hamiltonian at time t, shape (2, 2).
    """

    Ax = odeParams.drivingAmp * jnp.sin(odeParams.angularFreq * t)
    hx = jnp.sin(odeParams.kx - Ax)
    hy = jnp.sin(odeParams.ky)
    hz = odeParams.delta + jnp.cos(odeParams.ky - Ax) + jnp.cos(odeParams.ky)

    sigmax = jnp.array([[0, 1],
                        [1, 0]], dtype=jnp.complex64)

    sigmay = jnp.array([[0, -1j],
                        [1j, 0]], dtype=jnp.complex64)

    sigmaz = jnp.array([[1, 0],
                        [0, -1]], dtype=jnp.complex64) 
        
    return hx * sigmax + hy * sigmay + hz * sigmaz
    
# @jax.jit
def _band_basis_hamiltonian(t: float, p: DoubleTimeODEParams) -> jnp.ndarray[jnp.complex64]:
    """Rotates the lattice Hamiltonian into the band basis using precomputed U.

    Duplicate of a function found in the hamiltonian class, but that is necessary to be able
    utilise jax.

    parameters
    ----------
    t : float
        the time, in seconds, to evaluate the hamiltonian at.
    odeparams : doubletimeodeparams
        the relevant parameters for calculation of the hamiltonian.

    Returns
    -------
    ndarray[complex64]
        The Hamiltonian at time t in the band basis, shape (2, 2).
    """

    H_lattice = _lattice_basis_hamiltonian(t, p)
    return p.U.conj().T @ H_lattice @ p.U

# @jax.jit
def hm(t: float, p: DoubleTimeODEParams) -> complex:
    """Calculates the sigma_- coefficient of the band basis Hamiltonian.
        
    Duplicate of a function found in the hamiltonian class, but that is necessary to be able
    utilise jax.

    parameters
    ----------
    t : float
        the time, in seconds, to evaluate the hamiltonian at.
    odeparams : doubletimeodeparams
        the relevant parameters for calculation of the hamiltonian.

    Returns
    -------
    complex
        The sigma_- coefficient of the band basis Hamiltonian.
    """

    return _band_basis_hamiltonian(t, p)[1, 0]

# @jax.jit
def hp(t: float, p: DoubleTimeODEParams) -> complex:
    """Calculates the sigma_+ coefficient of the band basis Hamiltonian.
        
    Duplicate of a function found in the hamiltonian class, but that is necessary to be able
    utilise jax.

    parameters
    ----------
    t : float
        the time, in seconds, to evaluate the hamiltonian at.
    odeparams : doubletimeodeparams
        the relevant parameters for calculation of the hamiltonian.

    Returns
    -------
    complex
        The sigma_+ coefficient of the band basis Hamiltonian.
    """

    return _band_basis_hamiltonian(t, p)[0, 1]

# @jax.jit
def hz(t: float, p: DoubleTimeODEParams) -> complex:
    """Calculates the sigma_z coefficient of the band basis Hamiltonian.
        
    Duplicate of a function found in the hamiltonian class, but that is necessary to be able
    utilise jax.

    parameters
    ----------
    t : float
        the time, in seconds, to evaluate the hamiltonian at.
    odeparams : doubletimeodeparams
        the relevant parameters for calculation of the hamiltonian.

    Returns
    -------
    complex
        The sigma_z coefficient of the band basis Hamiltonian.
    """

    H = _band_basis_hamiltonian(t, p)
    return 0.5 * (H[0, 0] - H[1, 1])
    
# @jax.jit
def EquationsOfMotion(
        t: float,
        c: jnp.ndarray[complex],
        inhomPart: jnp.ndarray[complex],
        p: DoubleTimeODEParams
    ) -> np.ndarray[complex]:
    """
    Returns the right-hand side of the equations of motion for the system at time t, in seconds.

    Parameters
    ----------
    t : float | ndarray[float]
        The time, in seconds, at which to evaluate the equations of motion.
        Accepts vectorised inputs.
    c : ndarray[complex]
        The state of the system at time t, in the band basis.
        For single-time correlations, c should be (sigma_-, sigma_+, sigma_z).
    inhomPart : complex | ndarray[complex], shape(3,)
        The inhomogenous part of the equations of motion. This will change depending on whether we
        are solving for the single-time or double-time correlations.

        This can either be a scalar, in which case it is the z-component of the inhomogenous part, or
        a vector of shape (3,), in which case it forms the bottom row of the inhomogenous (3, 3) matrix.
    p: DoubleTimeODEParams
        The required parameters to call the double-time function with.

    Returns
    -------
    ndarray[complex]:
        The right hand side of the equations of motion. If the input is vectorised, the output
        will also be vectorised, with the first axis having size 3, corresponding to the different operators
        in c.
    """
        
    # For all left-operators and right-operators, the Hamiltonian is calculated
    # just as a function of t + tau. Hence, for all left- and right-operators,
    # the B matrix should be shared.

    # The most efficient this can be is calculating the B matrix once for all 9 correlations,
    # which occurs when we batch the ODE.
    Hm, Hp, Hz = hm(t, p), hp(t, p), hz(t, p)
    gamma = p.gamma

    B = jnp.array([[-(2j * Hz + 0.5 * gamma), 0, 1j * Hp],
                    [0, 2j * Hz - 0.5 * gamma, -1j * Hm],
                    [2j * Hm, -2j * Hp, -gamma]], dtype=jnp.complex64)
  
    # If the solution is not batched, just solve the ODE normally.
    if c.shape[0] == 3:
        inhomPartVector = jnp.array([0, 0, inhomPart], dtype=jnp.complex64)
        return B @ c + inhomPartVector[:, jnp.newaxis]
        
    # If the solution is batched, we basically have to solve 3 components simultaneously.
    if c.shape[0] == 9:
        # If we have batched, we will have formed a 3x3 array with the columns corresponding
        # to the same left-operator, which we would have then .ravel()-ed. Therefore, if we just
        # reshape (3, 3, -1) (with the last axis dealing with the fact that our input may be vectorised),
        # we will get back our stack of 3 x 3 matrices.
        c = c.reshape(3, 3, -1)
        # Moves the matrices stored in the first two axes to be stored in the last two axes,
        # which is the shape numpy expects when doing stacked matrix multiplication.
        c = jnp.moveaxis(c, [0, 1], [1, 2])

        # We will form our inhomogenous part from the inhomPart vector, which we will in this case assume has 3 components
        # which form the bottom row of our matrix.
        inhomPartMatrix = jnp.zeros((3, 3), dtype=jnp.complex64)
        inhomPartMatrix= inhomPartMatrix.at[:, 2].set(inhomPart)

        # We can now calculate our matrix multiplications.
        # Since B is simply (3, 3), numpy performs stacked matrix multiplication forming a final matrix of
        # (k, 3, 3), where k is due to the vectorised input.
        # Then, we add our inhomogenous part to each of those matrices in the stack.
        # This ends up being a (k, 3, 3) stack of matrices.
        dcdt = B @ c + inhomPartMatrix[jnp.newaxis, :, :]

        # Now, we reshape our matrix to be back in the shape (9, k), which is what scipy expects.
        return jnp.moveaxis(dcdt, [1, 2], [0, 1]).reshape(9, -1)