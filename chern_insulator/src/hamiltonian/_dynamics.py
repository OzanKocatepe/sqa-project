import numpy as np

class DynamicsMixin:
    """
    Contains all logic for the evolution of the operators.

    Methods
    -------
    EquationsOfMotion : The equation of motion for the Pauli operators. 
    """

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

        Hm, Hp, Hz = self.Hm(t), self.Hp(t), self.Hz(t)
        gamma = self.__params.decayConstant

        B = np.array([[-(2j * Hz + 0.5 * gamma), 0, 1j * Hp],
                      [0, 2j * Hz - 0.5 * gamma, -1j * Hm],
                      [2j * Hm, -2j * Hp, -gamma]], dtype=complex)
        
        inhomPart = np.array([0, 0, -gamma], dtype=complex)
 
        return B @ c + inhomPart[:, np.newaxis]