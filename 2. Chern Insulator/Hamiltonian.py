import numpy as np
from functools import cache, cached_property

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
        t : float | np.ndarray[float]
            The time, in seconds, at which to evaluate the Hamiltonian.
            Accepts vectorised inputs.

        Returns
        -------
        float | np.ndarray[float]:
            The value of the driving field in the x-direction at time(s) t.
            The type returned is the same as the type of t.
        """

        return self.__params.drivingAmplitude * np.sin(self.__params.angularFreq * t)

    def hx(self, t: float | np.ndarray[float]=0) -> float | np.ndarray[float]:
        """
        Returns the coefficient of sigma_x in the driven Hamiltonian at time t.
        
        Parameters
        ----------
        t : float | np.ndarray[float], optional
            The time, in seconds, at which to evaluate the Hamiltonian.
            Accepts vectorised inputs.
            If called without a time value (or with t = 0), returns the coefficient of
            sigma_x in the undriven Hamiltonian.

        Returns
        -------
        float | np.ndarray[float]:
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
    
    def hz(self, t: float | np.ndarrayp[float]=0) -> float | np.ndarray[float]:
        """
        Returns the coefficient of sigma_z in the driven Hamiltonian at time t.
        
        Parameters
        ----------
        t : float | np.ndarray[float], optional
            The time, in seconds, at which to evaluate the Hamiltonian.
            Accepts vectorised inputs.
            If called without a time value (or with t = 0), returns the coefficient of
            sigma_z in the undriven Hamiltonian.

        Returns
        -------
        float | np.ndarray[float]:
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
    
    def HMinus(self, t: float | np.ndarray[float]) -> float | np.ndarray[float]:
        """
        Returns the coefficient of sigma_- in the driven Hamiltonian in the band basis,
        at time t.

        Parameters
        ----------
        t : float | np.ndarray[float]
            The time, in seconds, at which to evaluate the Hamiltonian.
            Accepts vectorised inputs.

        Returns
        -------
        float | np.ndarray[float]:
            The value of the coefficient of sigma_- in the driven Hamiltonian in the band basis
            at time(s) t. The type returned is the same as the type of t.
        """

        rho = self.rho
        energy = self.energy()
        hx, hy, hz = self.hx(), self.hy(), self.hz()

        return self.hx(t) * (1j * hy - hx * hz / energy) / rho \
            -1j * self.hy(t) * (hx - 1j * hy * hz / energy) / rho \
            + self.hz(t) * rho / energy
    
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