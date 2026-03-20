import numpy as np
from functools import cache, cached_property
from scipy import special

from data import ModelParameters

class Hamiltonian:
    """Contains the functions derived from the Chern Hamiltonian."""

    def __init__(self, params: ModelParameters) -> None:
        """
        Initialises the Hamiltonian with the model parameters.
        
        Parameters
        ----------
        params : ModelParameters
            The parameters of the Chern insulator model at this momentum.
        """

        self.__params = params

    def Ax(self, t: float | np.ndarray[float]) -> float | np.ndarray[float]:
        """
        Returns the value of the driving field in the x-direction at time t.

        Parameters
        ----------
        t : float | ndarray[float]
            The time, in seconds, at which to evaluate the Hamiltonian.
            Accepts vectorised inputs.

        Returns
        -------
        float | ndarray[float]:
            The value of the driving field in the x-direction at time(s) t.
            The type returned is the same as the type of t.
        """

        return self.__params.drivingAmp * np.sin(self.__params.angularFreq * t)

    def hx(self, t: float | np.ndarray[float]=0) -> float | np.ndarray[float]:
        """
        Returns the coefficient of sigma_x in the driven Hamiltonian at time t.
        
        Parameters
        ----------
        t : float | ndarray[float], optional
            The time, in seconds, at which to evaluate the Hamiltonian.
            Accepts vectorised inputs.
            If called without a time value (or with t = 0), returns the coefficient of
            sigma_x in the undriven Hamiltonian.

        Returns
        -------
        float | ndarray[float]:
            The coefficient of sigma_x in the driven Hamiltonian at time(s) t.
            The type returned is the same as the type of t, if given.
            If no t is given, returns a float.
        """

        return np.sin(self.__params.kx - self.Ax(t))
    
    @cache   
    def hy(self) -> float:
        """
        Returns the coefficient of sigma_y in the Hamiltonian.
        This result is cached, since we are only driving in the x-direction and hence
        this component has no dependence on time, and the momentum doesn't change once given.
        
        Returns
        -------
        float:
            The coefficient of sigma_y in the Hamiltonian.
        """

        return np.sin(self.__params.ky)
    
    def hz(self, t: float | np.ndarray[float]=0) -> float | np.ndarray[float]:
        """
        Returns the coefficient of sigma_z in the driven Hamiltonian at time t.
        
        Parameters
        ----------
        t : float | ndarray[float], optional
            The time, in seconds, at which to evaluate the Hamiltonian.
            Accepts vectorised inputs.
            If called without a time value (or with t = 0), returns the coefficient of
            sigma_z in the undriven Hamiltonian.

        Returns
        -------
        float | ndarray[float]:
            The coefficient of sigma_z in the driven Hamiltonian at time(s) t.
            The type returned is the same as the type of t, if given.
            If no t is given, returns a float.
        """

        return self.__params.delta + np.cos(self.__params.kx - self.Ax(t)) + np.cos(self.__params.ky)
    
    @cache
    def energy(self) -> float:
        """
        Returns the unperturbed energy of the system at this momentum point.
        This result is cached, since the unperturbed energy has no dependence on time,
        and the momentum doesn't change once given.

        Returns
        -------
        float:
            The energy of the unperturbed system at this momentum.
        """

        return np.sqrt(self.hx()**2 + self.hy()**2 + self.hz()**2)

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

        rho = self.rho
        energy = self.energy()
        hx, hy, hz = self.hx(), self.hy(), self.hz()

        return self.hx(t) * (1j * hy - hx * hz / energy) / rho \
            -1j * hy * (hx - 1j * hy * hz / energy) / rho \
            + self.hz(t) * rho / energy
    
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

        rho = self.rho
        energy = self.energy()
        hx, hy, hz = self.hx(), self.hy(), self.hz()

        return -self.hx(t) * (1j * hy + hx * hz / energy) / rho \
            + 1j * hy * (hx + 1j * hy * hz / energy) / rho \
            + self.hz(t) * rho / energy
    
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

        energy = self.energy()
        hx, hy, hz = self.hx(), self.hy(), self.hz()

        return (self.hx(t) * hx + hy * hy + self.hz(t) * hz) / energy
    
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

        energy = self.energy()
        hx, hy, hz = self.hx(), self.hy(), self.hz()
        rho = self.rho

        return np.cos(self.__params.kx - self.Ax(t)) * (1j * hy - hx * hz / energy) / rho \
            - np.sin(self.__params.kx - self.Ax(t)) * rho / energy
    
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

        energy = self.energy()
        hx, hy, hz = self.hx(), self.hy(), self.hz()
        rho = self.rho

        return -np.cos(self.__params.kx - self.Ax(t)) * (1j * hy + hx * hz / energy) / rho \
            - np.sin(self.__params.kx - self.Ax(t)) * rho / energy
    
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

        energy = self.energy()
        hx, hy, hz = self.hx(), self.hy(), self.hz()

        return (np.cos(self.__params.kx - self.Ax(t)) * hx - np.sin(self.__params.kx - self.Ax(t)) * hz) / energy
    
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

        energy = self.energy()
        hx, hy, hz = self.hx(), self.hy(), self.hz()
        rho = self.rho

        return -1j * np.cos(self.__params.ky) * (hx - 1j * hy * hz / energy) / rho \
            - np.sin(self.__params.ky) * rho / energy
    
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

        energy = self.energy()
        hx, hy, hz = self.hx(), self.hy(), self.hz()
        rho = self.rho

        return 1j * np.cos(self.__params.ky) * (hx + 1j * hy * hz / energy) / rho \
            - np.sin(self.__params.ky) * rho / energy
    
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

        energy = self.energy()
        hx, hy, hz = self.hx(), self.hy(), self.hz()

        return (np.cos(self.__params.ky) * hy - np.sin(self.__params.ky) * hz) / energy
    
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

        rho = self.rho
        energy = self.energy()
        hx, hy, hz = self.hx(), self.hy(), self.hz()

        return self.hxn(n) * (1j * hy - hx * hz / energy) / rho \
            -1j * self.hyn(n) * (hx - 1j * hy * hz / energy) / rho \
            + self.hzn(n) * rho / energy

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

        rho = self.rho
        energy = self.energy()
        hx, hy, hz = self.hx(), self.hy(), self.hz()

        return -self.hxn(n) * (1j * hy + hx * hz / energy) / rho \
            + 1j * self.hyn(n) * (hx + 1j * hy * hz / energy) / rho \
            + self.hzn(n) * rho / energy

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

        energy = self.energy()
        hx, hy, hz = self.hx(), self.hy(), self.hz()

        return (self.hxn(n) * hx + self.hyn(n) * hy + self.hzn(n) * hz) / energy
    
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
                      [0, (2j * Hz - 0.5 * gamma), -1j * Hm],
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