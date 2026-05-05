import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

from data import ModelParameters, Fourier
from operators import Hamiltonian, ParamagneticCurrentX, ParamagneticCurrentY, DiamagneticCurrentX
from LengthGauge import LengthGauge

class CurrentSolver:
    """Contains the code for solving for single- and double- time currents."""

    def __init__(self, params: ModelParameters) -> None:
        """
        Initialises the current solver.

        Parameters
        ----------
        params : ModelParameters
            The parameters of the model we are solving the correlations for.
        """

        self.__params = params
        self.__hamiltonian = Hamiltonian(self.__params)
        self.__jpx = ParamagneticCurrentX(self.__params, self.__hamiltonian)
        self.__jpy = ParamagneticCurrentY(self.__params, self.__hamiltonian)
        self.__jdx = DiamagneticCurrentX(self.__params, self.__hamiltonian)

    def CalculateParamagneticCurrent(self, time: float | np.ndarray[float], fourierSeries: list[Fourier]) -> np.ndarray[complex]:
        """Calculates the paramagnetic current.
        
        Parameters
        ----------
        time : float | ndarray[float]
            The points in time, in seconds, to evaluate the paramagnetic current operator at.
        fourierSeries : list[Fourier]
            The list containing the Fourier series for sigma_-, sigma_+,
            and sigma_z, in that order.

        Returns
        -------
        ndarray[complex]:
            The value of the paramagnetic current operator at the corresponding times.
            Has shape (2, time.size), where the first dimension corresponds to the
            current in the x-dimension and y-dimension for indices 0 and 1 respectively.
        """

        current = np.zeros((2, time.size), dtype=complex)

        sigmam = fourierSeries[0].Evaluate(time)
        sigmap = fourierSeries[1].Evaluate(time)
        sigmaz = fourierSeries[2].Evaluate(time)
 
        current[0, :] = (
            self.__jpx.minus(time) * sigmam
            + self.__jpx.plus(time) * sigmap
            + self.__jpx.z(time) * sigmaz
        )
 
        current[1, :] = (
            self.__jpy.minus(time) * sigmam
            + self.__jpy.plus(time) * sigmap
            + self.__jpy.z(time) * sigmaz
        )
        
        return current
    
    def CalculateDiamagneticCurrent(self, time : float | np.ndarray[float], fourierSeries: list[Fourier]) -> np.ndarray[complex]:
        """Calculates the xx-diamagnetic current.
        
        Parameters
        ----------
        time : float | ndarray[float]
            The points in time, in seconds, to evaluate the diamagnetic current operator at.
        fourierSeries : list[Fourier]
            The list containing the Fourier series for sigma_-, sigma_+,
            and sigma_z, in that order.

        Returns
        -------
        ndarray[complex]:
            The value of the diamagnetic current operator at the corresponding times.
            Has shape (time.size,), since we only need the xx-component of the diamagnetic
            current in our case.
        """

        sigmam = fourierSeries[0].Evaluate(time)
        sigmap = fourierSeries[1].Evaluate(time)
        sigmaz = fourierSeries[2].Evaluate(time)
        
        return (
            self.__jdx.minus(time) * sigmam
            + self.__jdx.plus(time) * sigmap
            + self.__jdx.z(time) * sigmaz
        )
    
    def __SolveSigmaNumerically(self, time: np.ndarray[float]) -> np.ndarray[complex]:
        """
        TO BE DEPRECATED - if this continues to exist it must be moved and integrated
        properly into the code in another place.

        Solves the sigma correlations numerically as a debug step.

        Parameters
        ----------
        time : ndarray[float]
            The time, in seconds, that we will evaluate the solution at.
        
        Returns
        -------
        ndarray[complex]
            An array of shape (3, time.size) containing the evaluated sigma
            correlations.
        """

        raise DeprecationWarning("This function is deprecated.")

        return integrate.solve_ivp(
            fun = self.__hamiltonian.EquationsOfMotion,
            t_span = (0, np.max(time)),
            y0 = np.array([0.0, 0.0, -1.0], dtype=complex),
            t_eval = time,
            rtol=1e-9,
            atol=1e-12,
            vectorized = True
        ).y
    
    def CalculateLengthGaugeCurrent(self, time: float | np.ndarray[float]) -> np.ndarray[complex]:
        """Calculates the expectation value of the current in the length gauge.
        
        Parameters
        ----------
        time : float | ndarray[float]
            The points in time, in seconds, to evaluate the current operator at.

     
        Returns
        -------
        ndarray[complex]:
            The value of the length gauge current operator at the corresponding times.
            Has shape (2, time.size), where the first dimension corresponds to the
            current in the x-dimension and y-dimension for indices 0 and 1 respectively.
        """

        current = np.zeros((2, time.size), dtype=complex)

        lg = LengthGauge(self.__params, self.__hamiltonian)

        # Solves the ODE for our density matrix at the desired times.
        rho = integrate.solve_ivp(
            fun = lg.DensityMatrixODE,
            t_span = (0, np.max(time)),
            y0 = np.array([0.0, 0.0, 0.0, 1.0], dtype=complex),
            t_eval = time,
            rtol=1e-9,
            atol=1e-12
        ).y.T

        # Reshapes rho into a matrix rather than a flattened array.
        rho = rho.reshape((time.size, 2, 2))

        # Calculates our current operators at each desired time.
        jx = lg.jxLengthGauge(time)
        jy = lg.jyLengthGauge(time)

        # Calculates the average current.
        current[0, :] = np.trace(jx @ rho, axis1=1, axis2=2)
        current[1, :] = np.trace(jy @ rho, axis1=1, axis2=2)

        return current