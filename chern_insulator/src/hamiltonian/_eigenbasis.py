import numpy as np

class EigenbasisMixin:
    """
    Contains the logic for working with the Hamiltonian in its eigenbasis.

    Methods
    -------
    [DEPRECATED] Hpp: Gets the H++ component of the driven Hamiltonian in the band basis.
    [DEPRECATED] Hmm: Gets the H-- component of the driven Hamiltonian in the band basis.
    [DEPRECATED] Hpm: Gets the H+- component of the driven Hamiltonian in the band basis.
    [DEPRECATED] Hmp: Gets the H-+ component of the driven Hamiltonian in the band basis.
    Hm: Gets the coefficient of sigma_- for the driven Hamiltonian in the band basis.
    Hp: Gets the coefficient of sigma_+ for the driven Hamiltonian in the band basis.
    Hz: Gets the coefficient of sigma_z for the driven Hamiltonian in the band basis.
    """

    def Hpp(self, t: float | np.ndarray[float]) -> complex | np.ndarray[complex]:
        """
        Gets the H++ component of the Hamiltonian in the eigenbasis (band basis).

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
        res = (self.plusEigenvector.conj().T @ self.H(t) @ self.plusEigenvector).squeeze()

        if res.size == 1:
            return res.item()
        return res

    def Hmm(self, t: float | np.ndarray[float]) -> complex | np.ndarray[complex]:
        """
        Gets the H-- component of the Hamiltonian in the eigenbasis (band basis).

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
        res = (self.minusEigenvector.conj().T @ self.H(t) @ self.minusEigenvector).squeeze()

        if res.size == 1:
            return res.item()
        return res

    def Hpm(self, t: float | np.ndarray[float]) -> complex | np.ndarray[complex]:
        """
        Gets the H+- component of the Hamiltonian in the eigenbasis (band basis).

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
        res = (self.plusEigenvector.conj().T @ self.H(t) @ self.minusEigenvector).squeeze()

        if res.size == 1:
            return res.item()
        return res

    def Hmp(self, t: float | np.ndarray[float]) -> complex | np.ndarray[complex]:
        """
        Gets the H-+ component of the Hamiltonian in the eigenbasis (band basis).

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
        res = (self.minusEigenvector.conj().T @ self.H(t) @ self.plusEigenvector).squeeze()

        if res.size == 1:
            return res.item()
        return res

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

        self._GetMinus(self.H(t))
    
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