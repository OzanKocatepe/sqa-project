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
    Hmn: Calculates The Fourier series coefficients for the driven term H-(t).
    Hpn: Calculates The Fourier series coefficients for the driven term H+(t).
    Hzn: Calculates The Fourier series coefficients for the driven term Hz(t).
    """
 
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