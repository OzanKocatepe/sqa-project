import numpy as np
from functools import cache

class CurrentMixin:
    """
    Contains all of the required logic for working with the current in the lattice and band bases. 

    Methods:
    --------
    jpx: Calculates the paramagnetic x-current operator in the lattice basis at time t.
    jpy: Calculates the paramagnetic y-current operator in the lattice basis at time t.
    jpxm: Calculates the coefficient of sigma_- for the paramagnetic x-current in the band basis.
    jpxp: Calculates the coefficient of sigma_+ for the paramagnetic x-current in the band basis.
    jpxz: Calculates the coefficient of sigma_z for the paramagnetic x-current in the band basis.
    jpym: Calculates the coefficient of sigma_- for the paramagnetic y-current in the band basis.
    jpyp: Calculates the coefficient of sigma_+ for the paramagnetic y-current in the band basis.
    jpyz: Calculates the coefficient of sigma_z for the paramagnetic y-current in the band basis.
    """

    def jpx(self, t: float | np.ndarray[float]) -> np.ndarray[complex]:
        """
        Calculates the paramagnetic current operator in the x-direction
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

    def jpy(self) -> np.ndarray[complex]:
        """
        Calculates the paramagnetic current operator in the y-direction
        in the lattice basis.
         
        Returns
        -------
        np.ndarray[complex]
            The y-current operator in the lattice basis.
            Has shape (2, 2).
        """

        return -np.cos(self._params.ky) * self.sigmay + np.sin(self._params.ky) * self.sigmaz

    def jpxm(self, t: float | np.ndarray[float]) -> complex | np.ndarray[complex]:
        """
        Returns the coefficient of sigma_- for the paramagnetic current operator in the
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

        return self._GetMinus(self.jpx(t))
    
    def jpxp(self, t: float | np.ndarray[float]) -> complex | np.ndarray[complex]:
        """
        Returns the coefficient of sigma_+ for the paramagnetic current operator in the
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

        return self._GetPlus(self.jpx(t))
    
    def jpxz(self, t: float | np.ndarray[float]) -> float | np.ndarray[float]:
        """
        Returns the coefficient of sigma_z for the paramagnetic current operator in the
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

        return self._GetZ(self.jpx(t))
 
    @cache
    def jpym(self) -> complex:
        """
        Returns the coefficient of sigma_- for the paramagnetic current operator in the
        y-direction, in the band basis.

        Returns
        -------
        complex
            The coefficient of sigma_- for the current operator in the
            y-direction, in the band basis.
        """

        return self._GetMinus(self.jpy())
    
    @cache
    def jpyp(self) -> complex:
        """
        Returns the coefficient of sigma_+ for the paramagnetic current operator in the
        y-direction, in the band basis.

        Returns
        -------
        complex:
            The coefficient of sigma_+ for the current operator in the
            y-direction, in the band basis.
        """

        return self._GetPlus(self.jpy())
    
    @cache
    def jpyz(self) -> complex:
        """
        Returns the coefficient of sigma_z for the paramagnetic current operator in the
        y-direction, in the band basis.

        Returns
        -------
        ndarray[complex]:
            The coefficient of sigma_z for the current operator in the
            y-direction, in the band basis.
        """

        return self._GetZ(self.jpy())

    def jdxx(self, t: float | np.ndarray[float]) -> np.ndarray[complex]:
        """
        Calculates the xx diamagnetic current operator 
        in the lattice basis at some time t.
        
        Parameters
        ----------
        t : float | ndarray[float]
            The time, in seconds, at which to evaluate the current operator.
            Accepts vectorised inputs.
            
        Returns
        -------
        complex | ndarray[complex]
            The xx-current operator at time(s) t in the lattice basis. The type returned
            is the same type as t. Has shape (t.size, 2, 2)
        """

        t = np.atleast_1d(t)

        jxx = np.multiply.outer(-np.sin(self.__params.kx - self.Ax(t)), self.sigmax) \
            + np.multiply.outer(-np.cos(self._params.kx - self.Ax(t)), self.sigmaz)
        
        return jxx.squeeze()

    def jdyy(self) -> np.ndarray[complex]:
        """
        Calculates the yy diamagnetic current operator 
        in the lattice basis.
     
        Returns
        -------
        complex | ndarray[complex]
            The yy-current operator in the lattice basis. Has shape (2, 2).
        """

        t = np.atleast_1d(t)

        return -np.sin(self.__params.ky) * self.sigmay - np.cos(self.__params.ky) * self.sigmaz 