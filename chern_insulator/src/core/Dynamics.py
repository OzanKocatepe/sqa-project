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
                          c: np.ndarray[complex]
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

        Returns
        -------
        ndarray[complex]:
            The right hand side of the equations of motion. If the input is vectorised, the output
            will also be vectorised, with the first axis having size 3, corresponding to the different operators
            in c.
        """

        Hm, Hp, Hz = self.ham.minus(t), self.ham.plus(t), self.ham.z(t)
        gamma = self._params.decayConstant

        B = np.array([[-(2j * Hz + 0.5 * gamma), 0, 1j * Hp],
                      [0, 2j * Hz - 0.5 * gamma, -1j * Hm],
                      [2j * Hm, -2j * Hp, -gamma]], dtype=complex)
        
        inhomPart = np.array([0, 0, -gamma], dtype=complex)
 
        return B @ c + inhomPart[:, np.newaxis]