import numpy as np

from data import ModelParameters
from operators import Hamiltonian

class Dynamics:
    """Stores all information about dynamics of the system - specifically the
    equations of motion for single- and double-time correlations.
    """

    def __init__(self, params: ModelParameters, hamiltonian: Hamiltonian):
        """Initialises the instance.
        
        Parameters
        ----------
        params : ModelParameters
            The parameters of the model.
        hamiltonian : Hamiltonian
            The Hamiltonian of the system, which is used to compute
            the time-dependent coefficients in the equations of motion.
        """

        self._params = params
        self._ham = hamiltonian
    
    def EquationsOfMotion(self, t: float | np.ndarray[float],
                          c: np.ndarray[complex],
                          inhomPart: complex | np.ndarray[complex],
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
        Hm, Hp, Hz = self._ham.minus(t), self._ham.plus(t), self._ham.z(t)
        gamma = self._params.decayConstant

        B = np.array([[-(2j * Hz + 0.5 * gamma), 0, 1j * Hp],
                      [0, 2j * Hz - 0.5 * gamma, -1j * Hm],
                      [2j * Hm, -2j * Hp, -gamma]], dtype=complex)
        
        # If the solution is not batched, just solve the ODE normally.
        if c.shape[0] == 3:
            inhomPartVector = np.array([0, 0, inhomPart], dtype=complex)
            return B @ c + inhomPartVector[:, np.newaxis]
        
        # If the solution is batched, we basically have to solve 3 components simultaneously.
        if c.shape[0] == 9:
            # If we have batched, we will have formed a 3x3 array with the columns corresponding
            # to the same left-operator, which we would have then .ravel()-ed. Therefore, if we just
            # reshape (3, 3, -1) (with the last axis dealing with the fact that our input may be vectorised),
            # we will get back our stack of 3 x 3 matrices.
            c = c.reshape(3, 3, -1)
            # Moves the matrices stored in the first two axes to be stored in the last two axes,
            # which is the shape numpy expects when doing stacked matrix multiplication.
            c = np.moveaxis(c, [0, 1], [1, 2])

            # We will form our inhomogenous part from the inhomPart vector, which we will in this case assume has 3 components
            # which form the bottom row of our matrix.
            inhomPartMatrix = np.zeros((3, 3), dtype=complex)

            # We can now calculate our matrix multiplications.
            # Since B is simply (3, 3), numpy performs stacked matrix multiplication forming a final matrix of
            # (k, 3, 3), where k is due to the vectorised input.

            # Then, we add our inhomogenous part to each of those matrices in the stack.
            # This ends up being a (k, 3, 3) stack of matrices.
            dcdt = B @ c + inhomPartMatrix[np.newaxis, :, :]

            # Now, we reshape our matrix to be back in the shape (9, k), which is what scipy expects.
            return np.moveaxis(dcdt, [1, 2], [0, 1]).reshape(9, -1)