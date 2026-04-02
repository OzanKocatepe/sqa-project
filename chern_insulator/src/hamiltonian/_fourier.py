import numpy as np
from scipy import special

class FourierMixin:
    """
    Contains all of the logic for working with the Hamiltonian in the lattice and band basis
    as a Fourier series.

    Methods
    -------
    hxn: Calculates the Fourier series coefficients for the driven term hx(t).
    hyn: Calculates the Fourier series coefficients for the driven term hy(t).
    hzn: Calculates the Fourier series coefficients for the driven term hz(t).
    Hn: Calculates the Fourier series coefficients for the driven Hamiltonian in the lattice basis.
    [DEPRECATED] Hppn: Calculates the Fourier series coefficients for the driven term H++(t).
    [DEPRECATED] Hmmn: Calculates the Fourier series coefficients for the driven term H--(t).
    [DEPRECATED] Hpmn: Calculates the Fourier series coefficients for the driven term H+-(t).
    [DEPRECATED] Hmpn: Calculates the Fourier series coefficients for the driven term H-+(t).
    Hmn: Calculates The Fourier series coefficients for the driven term H-(t).
    Hpn: Calculates The Fourier series coefficients for the driven term H+(t).
    Hzn: Calculates The Fourier series coefficients for the driven term Hz(t).
    """

    def hxn(self, n: int | np.ndarray[int]) -> complex | np.ndarray[complex]:
        """
        Calculates the nth Fourier coefficient for the driven term hx(t).

        Parameters
        ----------
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
        coeffs[oddMask] = np.sign(n[oddMask]) * 1j * special.jv(np.abs(n[oddMask]), self.__params.drivingAmp) * np.cos(self.__params.kx)
        coeffs[zeroMask] = special.jv(0, self.__params.drivingAmp) * np.sin(self.__params.kx)
        coeffs[evenMask] = special.jv(np.abs(n[evenMask]), self.__params.drivingAmp) * np.sin(self.__params.kx)

        # Returns the array as a float if it has size 1.
        if coeffs.size == 1:
            return coeffs[0]

        return coeffs
    
    def hyn(self, n: int | np.ndarray[int]) -> complex | np.ndarray[complex]:
        """
        Calculates the nth Fourier coefficient for the term hyn.
        This is a constant term, so returning the 'Fourier coefficients' just
        returns the value itself for n = 0, and 0 for all other n.
        This function is just a utility to use for compatibility with the other Fourier
        coefficient functions.

        Parameters
        ----------
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
        coeffs[n == 0] = self.hy()

        # Returns the array as a float if it has size 1.
        if coeffs.size == 1:
            return coeffs[0]

        return coeffs


    def hzn(self, n: int | np.ndarray[int]) -> complex | np.ndarray[complex]:
        """
        Calculates the nth Fourier coefficient for the driven term hz(t).

        Parameters
        ----------
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
        coeffs[evenMask] = special.jv(np.abs(n[evenMask]), self.__params.drivingAmp) * np.cos(self.__params.kx)
        coeffs[oddMask] = np.sign(n[oddMask]) * -1j * special.jv(np.abs(n[oddMask]), self.__params.drivingAmp) * np.sin(self.__params.kx)
        coeffs[zeroMask] = self.__params.delta + np.cos(self.__params.kx) * special.jv(0, self.__params.drivingAmp) \
                    + np.cos(self.__params.ky)

        # Returns the array as a float if it has size 1.
        if coeffs.size == 1:
            return coeffs[0]

        return coeffs
    
    def Hn(self, n: int | np.ndarray[int]) -> np.ndarray[complex]:
        """
        Calculates the nth Fourier coefficient for H(t), the driven lattice basis Hamiltonian.
        
        Parameters
        ----------
        n : int | ndarray[int]
            The index, or indices, that we will calculate the Fourier coefficients of.
            The index corresponds to the harmonic of the base frequency.

        Returns
        -------
        complex | ndarray[complex]:
            The desired coefficient(s). Of the same type as n. Comes in the shape
            (n.size, 2, 2), where the third axis disappears if n is a scalar.
        """

        return np.moveaxis(np.array([[self.hzn(n), self.hxn(n) - 1j * self.hyn(n)],
                         [self.hxn(n) + 1j * self.hyn(n), -self.hzn(n)]], dtype=complex),
                         -1, 0)
     
    def Hppn(self, n: int | np.ndarray[int]) -> complex | np.ndarray[complex]:
        """
        Gets the H++ component of the Hamiltonian in the eigenbasis (band basis).

        Parameters
        ----------
        n : int | ndarray[int]
            The components of the Fourier series to find the coefficients for.
            Can be vectorised, resulting in a vectorised output.
        
        Returns
        ----------
        complex | ndarray[complex]:
            The desired coefficient.
            The type returned is the same as the type of n.
        """

        raise DeprecationWarning()

        # We squeeze the result since if t has size (n,), the result will have size (n, 1).
        res = (self.plusEigenvector.conj().T @ self.Hn(n) @ self.plusEigenvector).squeeze()

        if res.size == 1:
            return res.item()
        return res

    def Hmmn(self, n: int | np.ndarray[int]) -> complex | np.ndarray[complex]:
        """
        Gets the H-- component of the Hamiltonian in the eigenbasis (band basis).

        Parameters
        ----------
        n : int | ndarray[int]
            The components of the Fourier series to find the coefficients for.
            Can be vectorised, resulting in a vectorised output.
        
        Returns
        ----------
        complex | ndarray[complex]:
            The desired coefficient.
            The type returned is the same as the type of n.
        """

        raise DeprecationWarning()

        # We squeeze the result since if t has size (n,), the result will have size (n, 1).
        res = (self.minusEigenvector.conj().T @ self.Hn(n) @ self.minusEigenvector).squeeze()

        if res.size == 1:
            return res.item()
        return res

    def Hpmn(self, n: int | np.ndarray[int]) -> complex | np.ndarray[complex]:
        """
        Gets the H+- component of the Hamiltonian in the eigenbasis (band basis).

        Parameters
        ----------
        n : int | ndarray[int]
            The components of the Fourier series to find the coefficients for.
            Can be vectorised, resulting in a vectorised output.
        
        Returns
        ----------
        complex | ndarray[complex]:
            The desired coefficient.
            The type returned is the same as the type of n.
        """

        raise DeprecationWarning()

        # We squeeze the result since if t has size (n,), the result will have size (n, 1).
        res = (self.plusEigenvector.conj().T @ self.Hn(n) @ self.minusEigenvector).squeeze()

        if res.size == 1:
            return res.item()
        return res

    def Hmpn(self, n: int | np.ndarray[int]) -> complex | np.ndarray[complex]:
        """
        Gets the H-+ component of the Hamiltonian in the eigenbasis (band basis).

        Parameters
        ----------
        n : int | ndarray[int]
            The components of the Fourier series to find the coefficients for.
            Can be vectorised, resulting in a vectorised output.
        
        Returns
        ----------
        complex | ndarray[complex]:
            The desired coefficient.
            The type returned is the same as the type of n.
        """

        raise DeprecationWarning()

        # We squeeze the result since if t has size (n,), the result will have size (n, 1).
        res = (self.minusEigenvector.conj().T @ self.Hn(n) @ self.plusEigenvector).squeeze()

        if res.size == 1:
            return res.item()
        return res

    def Hmn(self, n: int | np.ndarray[int]) -> complex | np.ndarray[complex]:
        """
        Returns the corresponding coefficient in the Fourier series for H-(t).

        Parameters
        ----------
        n : int | ndarray[int]
            The components of the Fourier series to find the coefficients for.
            Can be vectorised, resulting in a vectorised output.

        Returns
        -------
        complex | ndarray[complex]:
            The desired coefficients, corresponding to the entries of n.
            The type returned is the same as the type of n.
        """

        return self._GetMinus(self.Hn(n))
    
    def Hpn(self, n: int | np.ndarray[int]) -> complex | np.ndarray[complex]:
        """
        Returns the corresponding coefficient in the Fourier series for H+(t).

        Parameters
        ----------
        n : int | ndarray[int]
            The components of the Fourier series to find the coefficients for.
            Can be vectorised, resulting in a vectorised output.

        Returns
        -------
        complex | ndarray[complex]:
            The desired coefficients, corresponding to the entries of n.
            The type returned is the same as the type of n.
        """

        return self._GetPlus(self.Hn(n))
    
    def Hzn(self, n : int | np.ndarray[int]) -> complex | np.ndarray[complex]:
        """
        Returns the corresponding coefficient in the Fourier series for Hz(t).

        Parameters
        ----------
        n : int | ndarray[int]
            The components of the Fourier series to find the coefficients for.
            Can be vectorised, resulting in a vectorised output.

        Returns
        -------
        complex | ndarray[complex]:
            The desired coefficients, corresponding to the entries of n.
            The type returned is the same as the type of n.
        """

        return self._GetZ(self.Hn(n))