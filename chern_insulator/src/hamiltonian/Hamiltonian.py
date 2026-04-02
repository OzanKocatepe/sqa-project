import numpy as np
from functools import cache, cached_property
from scipy import special

from data import ModelParameters

class Hamiltonian:
    """Contains the functions derived from the Chern Hamiltonian."""
  
    @staticmethod
    def StaticEnergy(kx: float | np.ndarray[float], ky: float | np.ndarray[float], delta: float) -> float | np.ndarray[float]:
        """
        Exact same thing as energy, but it turns out to be useful to have a
        static instance of this function so that we can calculate
        energies throughout the Brillouin zone without relying on other properties.

        We also vectorise this function in the momentums to speed things up.

        Parameters
        ----------
        kx : float | ndarray[float]
            The x-component of the momentum.
        ky : float | ndarray[float]
            The x-component of the momentum.
            Must be the same type as kx.
        delta : float
            The value of delta.

        Returns
        -------
        float | ndarray[float]:
            The energy of the unperturbed system at this momentum
            and delta.
            Will be the same type as kx and ky.
        """

        hx = np.sin(kx)
        hy = np.sin(ky)
        hz = delta + np.cos(kx) + np.cos(ky)

        return np.sqrt(hx**2 + hy**2 + hz**2) 
      
    
    # @cache
    def EigenbasisComponentsFourier(self, n: int | np.ndarray[int]) -> np.ndarray[complex]:
        """
        Calculates the nth Fourier coefficient for the components of the Hamiltonian in the eigenbasis (band basis).
        Uses no numerical approximations.

        Parameters
        ----------
        n : int | ndarray[int]
            The index, or indices, that we will calculate the Fourier coefficients of.
            The index corresponds to the harmonic of the base frequency. Vectorisable.
        
        Returns
        ----------
        ndarray[complex]:
            The components in matrix form, in the form
            (H++, H+-,
             H-+, H--)
            for the nth Fourier coefficient. If n is a scalar, this is just a (2, 2) array.
            If n is a vector, this is an array of shape (n.size, 2, 2).
        """

        n = np.atleast_1d(n)
        coeffs = np.zeros((n.size, 2, 2), dtype=complex)

        # Gets the diagonal components.
        coeffs[:, 0, 0] = np.trace(self.Hn(n) @ self.PPlus(), axis1=1, axis2=2)
        coeffs[:, 1, 1] = np.trace(self.Hn(n) @ self.PMinus(), axis1=1, axis2=2)

        # Defines our arbitrary vector numerically.
        # Numerical uncertainty in this should not cause numerical uncertainty in our final values,
        # since this will work for any arbitrary vector not orthogonal to either eigenvector.
        _, U = np.linalg.eigh(self.H())
        r = (U[:, 0] + U[:, 1]).reshape(2, 1) / np.sqrt(2)

        # Calculates the off-diagonal components.
        # denominator = np.sqrt( r.conj().T @ self.PPlus() @ r @ r.conj().T @ self.PMinus() @ r )
        # coeffs[:, 0, 1] = ( r.conj().T @ self.PPlus() @ self.Hn(n) @ self.PMinus() @ r )[:, 0, 0] / denominator
        # coeffs[:, 1, 0] = ( r.conj().T @ self.PMinus() @ self.Hn(n) @ self.PPlus() @ r )[:, 0, 0] / denominator

        # coeffs[:, 0, 1] = (U[:, 0].reshape(2, 1).conj().T @ self.Hn(n) @ U[:, 1].reshape(2, 1))[:, 0, 0]
        # coeffs[:, 1, 0] = (U[:, 1].reshape(2, 1).conj().T @ self.Hn(n) @ U[:, 0].reshape(2, 1))[:, 0, 0]

        for index in range(coeffs.shape[0]):
            coeffs[index, 0, 1] = (U[:, 0].reshape(2, 1).conj().T @ self.Hn(n[index]) @ U[:, 1].reshape(2, 1))[0, 0]
            coeffs[index, 1, 0] = (U[:, 1].reshape(2, 1).conj().T @ self.Hn(n[index]) @ U[:, 0].reshape(2, 1))[0, 0]
        
        return coeffs
    
    def Hmn(self, n: int | np.ndarray[int]) -> complex | np.ndarray[complex]:
        """
        Calculates the nth Fourier coefficient for Hm(t).
        
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

        return self.EigenbasisComponentsFourier(n)[:, 1, 0]

    def Hpn(self, n: int | np.ndarray[int]) -> complex | np.ndarray[complex]:
        """
        Calculates the nth Fourier coefficient for Hp(t).
        
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

        return self.EigenbasisComponentsFourier(n)[:, 0, 1]

    def Hzn(self, n: int | np.ndarray[int]) -> complex | np.ndarray[complex]:
        """
        Calculates the nth Fourier coefficient for Hz(t).
        
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

        components = self.EigenbasisComponentsFourier(n)

        return 0.5 * (components[:, 0, 0] + components[:, 1, 1])
    
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
    
    
    @cached_property
    def rho(self) -> float:
        """
        Returns the value of rho, defined as sqrt(hx^2 + hy^2) where hx and hy are undriven terms.
        This is a commonly used term in the calculations.
        This result is cached, since rho has no time dependence.

        Returns
        -------
        float:
            The value of rho at this momentum point.
        """

        return np.sqrt(self.hx()**2 + self.hy()**2)