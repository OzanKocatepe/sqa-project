import numpy as np

class EigenbasisMixin:
    """
    Contains the logic for working with the Hamiltonian in its eigenbasis.

    Methods
    -------
    Hm: Gets the coefficient of sigma_- for the driven Hamiltonian in the band basis.
    Hp: Gets the coefficient of sigma_+ for the driven Hamiltonian in the band basis.
    Hz: Gets the coefficient of sigma_z for the driven Hamiltonian in the band basis.
    """

    def Hm(self, t: float | np.ndarray[float]) -> complex | np.ndarray[complex]:
        """
        Returns the coefficient of sigma_- in the driven Hamiltonian in the band basis,
        at time t.

        Parameters
        ----------
        t : float | ndarray[float]
            The time, in seconds, at which to evaluate the Hamiltonian.
            Accepts vectorised inputs.

        Returns
        -------
        complex | ndarray[complex]:
            The value of the coefficient of sigma_- in the driven Hamiltonian in the band basis
            at time(s) t. The type returned is the same as the type of t.
        """

        return self._GetMinus(self.H(t))
    
    def Hp(self, t: float | np.ndarray[float]) -> complex | np.ndarray[complex]:
        """
        Returns the coefficient of sigma_+ in the driven Hamiltonian in the band basis,
        at time t.

        Parameters
        ----------
        t : float | ndarray[float]
            The time, in seconds, at which to evaluate the Hamiltonian.
            Accepts vectorised inputs.

        Returns
        -------
        complex | ndarray[complex]
            The value of the coefficient of sigma_+ in the driven Hamiltonian in the band basis
            at time(s) t. The type returned is the same as the type of t.
        """

        return self._GetPlus(self.H(t))
    
    def Hz(self, t: float | np.ndarray[float]) -> complex | np.ndarray[complex]:
        """
        Returns the coefficient of sigma_z in the driven Hamiltonian in the band basis,
        at time t.

        Parameters
        ----------
        t : float | ndarray[float]
            The time, in seconds, at which to evaluate the Hamiltonian.
            Accepts vectorised inputs.

        Returns
        -------
        complex | ndarray[complex]:
            The value of the coefficient of sigma_z in the driven Hamiltonian in the band basis
            at time(s) t. The type returned is the same as the type of t.
        """

        return self._GetZ(self.H(t))

    def HI(self, t: float | np.ndarray[float]) -> complex | np.ndarray[complex]:
        """
        Returns the coefficient of the identity matrix in the driven Hamiltonian in the band basis,
        at time t.

        Parameters
        ----------
        t : float | ndarray[float]
            The time, in seconds, at which to evaluate the Hamiltonian.
            Accepts vectorised inputs.

        Returns
        -------
        complex | ndarray[complex]:
            The value of the coefficient of the identity matrix in the driven Hamiltonian in the band basis
            at time(s) t. The type returned is the same as the type of t.
        """

        return self._GetI(self.H(t))