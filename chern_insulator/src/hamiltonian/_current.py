import numpy as np
from functools import cache

class CurrentMixin:
    """
    Contains all of the required logic for working with the current in the lattice and band bases. 

    Methods:
    --------
    jx: Calculates the x-current operator in the lattice basis at time t.
    jy: Calculates the y-current operator in the lattice basis at time t.
    jxm: Calculates the coefficient of sigma_- for the x-current in the band basis.
    jxp: Calculates the coefficient of sigma_+ for the x-current in the band basis.
    jxz: Calculates the coefficient of sigma_z for the x-current in the band basis.
    jym: Calculates the coefficient of sigma_- for the y-current in the band basis.
    jyp: Calculates the coefficient of sigma_+ for the y-current in the band basis.
    jyz: Calculates the coefficient of sigma_z for the y-current in the band basis.
    """

    def jx(self, t: float | np.ndarray[float]) -> np.ndarray[complex]:
        """
        Calculates the current operator in the x-direction
        in the lattice basis at some time t.
        
        Parameters
        ----------
        t : float | ndarray[float]
            The time, in seconds, at which to evaluate the current operator.
            Accepts vectorised inputs.
            
        Returns
        -------
        complex | ndarray[complex]
            The x-current operator at time(s) t in the lattice basis. The type returned
            is the same type as t. Has shape (t.size, 2, 2)
        """

        t = np.atleast_1d(t)

        jx = np.multiply.outer(-np.cos(self._params.kx - self.Ax(t)), self.sigmax) \
            + np.multiply.outer(np.sin(self._params.kx - self.Ax(t)), self.sigmaz)
        
        return jx.squeeze()

    def jy(self) -> np.ndarray[complex]:
        """
        Calculates the current operator in the y-direction
        in the lattice basis.
         
        Returns
        -------
        np.ndarray[complex]
            The y-current operator at time(s) t in the lattice basis.
            Has shape (2, 2).
        """

        return -np.cos(self._params.ky) * self.sigmay + np.sin(self._params.ky) * self.sigmaz

    def jxm(self, t: float | np.ndarray[float]) -> complex | np.ndarray[complex]:
        """
        Returns the coefficient of sigma_- for the current operator in the
        x-direction, in the band basis, at time t.

        Parameters
        ----------
        t : float | ndarray[float]
            The time, in seconds, at which to evaluate the Hamiltonian.
            Accepts vectorised inputs.

        Returns
        -------
        complex | ndarray[complex]:
            The coefficient of sigma_- for the current operator in the
            x-direction, in the band basis, at time t.
            The type returned is the same as the type of t.
        """

        return self._GetMinus(self.jx(t))
    
    def jxp(self, t: float | np.ndarray[float]) -> complex | np.ndarray[complex]:
        """
        Returns the coefficient of sigma_+ for the current operator in the
        x-direction, in the band basis, at time t.

        Parameters
        ----------
        t : float | ndarray[float]
            The time, in seconds, at which to evaluate the Hamiltonian.
            Accepts vectorised inputs.

        Returns
        -------
        complex | ndarray[complex]
            The coefficient of sigma_+ for the current operator in the
            x-direction, in the band basis, at time t.
            The type returned is the same as the type of t.
        """

        return self._GetPlus(self.jx(t))
    
    def jxz(self, t: float | np.ndarray[float]) -> float | np.ndarray[float]:
        """
        Returns the coefficient of sigma_z for the current operator in the
        x-direction, in the band basis, at time t.

        Parameters
        ----------
        t : float | ndarray[float]
            The time, in seconds, at which to evaluate the Hamiltonian.
            Accepts vectorised inputs.

        Returns
        -------
        float | ndarray[float]:
            The coefficient of sigma_z for the current operator in the
            x-direction, in the band basis, at time t.
            The type returned is the same as the type of t.
        """

        return self._GetZ(self.jx(t))

    def jxi(self, t: float | np.ndarray[float]) -> float | np.ndarray[float]:
        """
        Returns the coefficient of the identity matrix for the current operator in the
        x-direction, in the band basis, at time t.

        Parameters
        ----------
        t : float | ndarray[float]
            The time, in seconds, at which to evaluate the Hamiltonian.
            Accepts vectorised inputs.

        Returns
        -------
        float | ndarray[float]:
            The coefficient of the identity matrix for the current operator in the
            x-direction, in the band basis, at time t.
            The type returned is the same as the type of t.
        """

        return self._GetI(self.jx(t))
    
    @cache
    def jym(self) -> complex:
        """
        Returns the coefficient of sigma_- for the current operator in the
        y-direction, in the band basis.

        Returns
        -------
        complex
            The coefficient of sigma_- for the current operator in the
            y-direction, in the band basis.
        """

        return self._GetMinus(self.jy())
    
    @cache
    def jyp(self) -> complex:
        """
        Returns the coefficient of sigma_+ for the current operator in the
        y-direction, in the band basis.

        Returns
        -------
        complex:
            The coefficient of sigma_+ for the current operator in the
            y-direction, in the band basis.
        """

        return self._GetPlus(self.jy())
    
    @cache
    def jyz(self) -> complex:
        """
        Returns the coefficient of sigma_z for the current operator in the
        y-direction, in the band basis.

        Returns
        -------
        ndarray[complex]:
            The coefficient of sigma_z for the current operator in the
            y-direction, in the band basis.
        """

        return self._GetZ(self.jy())

    @cache
    def jyi(self) -> complex:
        """
        Returns the coefficient of the identity matrix for the current operator in the
        y-direction, in the band basis.

        Returns
        -------
        ndarray[complex]:
            The coefficient of the identity matrix for the current operator in the
            y-direction, in the band basis.
        """

        return self._GetI(self.jy())