import numpy as np
from functools import cache

class CurrentMixin:
    """
    Contains all of the required logic for working with the current in the lattice and band bases. 

    Methods:
    --------
    jx: Calculates the x-current operator in the lattice basis at time t.
    jy: Calculates the y-current operator in the lattice basis at time t.
    [DEPRECATED] jxpp: Calculates the jx++ component of the x-current in the band basis.
    [DEPRECATED] jxmm: Calculates the jx-- component of the x-current in the band basis.
    [DEPRECATED] jxpm: Calculates the jx+- component of the x-current in the band basis.
    [DEPRECATED] jxmp: Calculates the jx-+ component of the x-current in the band basis.
    [DEPRECATED] jypp: Calculates the jy++ component of the y-current in the band basis.
    [DEPRECATED] jymm: Calculates the jy-- component of the y-current in the band basis.
    [DEPRECATED] jypm: Calculates the jy+- component of the y-current in the band basis.
    [DEPRECATED] jymp: Calculates the jy-+ component of the y-current in the band basis.
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

        jx = -np.cos(self.__params.kx - self.Ax(t))[:, np.newaxis, np.newaxis] * self.sigmax \
            + np.sin(self.__params.kx - self.Ax(t))[:, np.newaxis, np.newaxis] * self.sigmaz
        
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

        return -np.cos(self.__params.ky) * self.sigmay + np.sin(self.__params.ky) * self.sigmaz

    def jxpp(self, t: float | np.ndarray[float]) -> complex | np.ndarray[complex]:
        """
        Gets the j++ component of the current operator in the x-direction in the band basis.

        Parameters
        ----------
        t : float | ndarray[float]
            The time, in seconds, to find the component.
            Can be vectorised, resulting in a vectorised output.
        
        Returns
        ----------
        complex | ndarray[complex]:
            The desired component at time(s) t.
            The type returned is the same as the type of t.
        """

        raise DeprecationWarning()

        # We squeeze the result since if t has size (n,), the result will have size (n, 1).
        res = (self.plusEigenvector.conj().T @ self.jx(t) @ self.plusEigenvector).squeeze()

        if res.size == 1:
            return res.item()
        return res

    def jxmm(self, t: float | np.ndarray[float]) -> complex | np.ndarray[complex]:
        """
        Gets the j-- component of the current operator in the x-direction in the band basis.

        Parameters
        ----------
        t : float | ndarray[float]
            The time, in seconds, to find the component.
            Can be vectorised, resulting in a vectorised output.
        
        Returns
        ----------
        complex | ndarray[complex]:
            The desired component at time(s) t.
            The type returned is the same as the type of t.
        """

        raise DeprecationWarning()

        # We squeeze the result since if t has size (n,), the result will have size (n, 1).
        res = (self.minusEigenvector.conj().T @ self.jx(t) @ self.minusEigenvector).squeeze()

        if res.size == 1:
            return res.item()
        return res

    def jxpm(self, t: float | np.ndarray[float]) -> complex | np.ndarray[complex]:
        """
        Gets the j+- component of the current operator in the x-direction in the band basis.

        Parameters
        ----------
        t : float | ndarray[float]
            The time, in seconds, to find the component.
            Can be vectorised, resulting in a vectorised output.
        
        Returns
        ----------
        complex | ndarray[complex]:
            The desired component at time(s) t.
            The type returned is the same as the type of t.
        """

        raise DeprecationWarning()

        # We squeeze the result since if t has size (n,), the result will have size (n, 1).
        res = (self.plusEigenvector.conj().T @ self.jx(t) @ self.minusEigenvector).squeeze()

        if res.size == 1:
            return res.item()
        return res
    
    def jxmp(self, t: float | np.ndarray[float]) -> complex | np.ndarray[complex]:
        """
        Gets the j-+ component of the current operator in the x-direction in the band basis.

        Parameters
        ----------
        t : float | ndarray[float]
            The time, in seconds, to find the component.
            Can be vectorised, resulting in a vectorised output.
        
        Returns
        ----------
        complex | ndarray[complex]:
            The desired component at time(s) t.
            The type returned is the same as the type of t.
        """

        raise DeprecationWarning()

        # We squeeze the result since if t has size (n,), the result will have size (n, 1).
        res = (self.minusEigenvector.conj().T @ self.jx(t) @ self.plusEigenvector).squeeze()

        if res.size == 1:
            return res.item()
        return res

    @cache
    def jypp(self) -> complex:
        """
        Gets the j++ component of the current operator in the y-direction in the band basis.
 
        Returns
        ----------
        complex
            The desired component.
        """

        raise DeprecationWarning()

        # We squeeze the result since if t has size (n,), the result will have size (n, 1).
        return (self.plusEigenvector.conj().T @ self.jy() @ self.plusEigenvector).item()

    @cache
    def jymm(self) -> complex:
        """
        Gets the j-- component of the current operator in the y-direction in the band basis.
 
        Returns
        ----------
        complex
            The desired component.
        """

        raise DeprecationWarning()

        # We squeeze the result since if t has size (n,), the result will have size (n, 1).
        return (self.minusEigenvector.conj().T @ self.jy() @ self.minusEigenvector).item()

    @cache
    def jypm(self) -> complex:
        """
        Gets the j+- component of the current operator in the y-direction in the band basis.
 
        Returns
        ----------
        complex
            The desired component.
        """

        raise DeprecationWarning()

        # We squeeze the result since if t has size (n,), the result will have size (n, 1).
        return (self.plusEigenvector.conj().T @ self.jy() @ self.minusEigenvector).item()
    
    @cache
    def jymp(self) -> complex:
        """
        Gets the j-+ component of the current operator in the y-direction in the band basis.
 
        Returns
        ----------
        complex
            The desired component.
        """

        raise DeprecationWarning()

        # We squeeze the result since if t has size (n,), the result will have size (n, 1).
        return (self.minusEigenvector.conj().T @ self.jy() @ self.plusEigenvector).item()

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