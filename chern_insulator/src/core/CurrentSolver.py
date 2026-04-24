import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

from data import ModelParameters, Fourier
from hamiltonian import Hamiltonian

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
 
        current[0, :] = self.__hamiltonian.jpxm(time) * sigmam \
            + self.__hamiltonian.jpxp(time) * sigmap \
            + self.__hamiltonian.jpxz(time) * sigmaz
 
        current[1, :] = self.__hamiltonian.jpym() * sigmam \
            + self.__hamiltonian.jpyp() * sigmap \
            + self.__hamiltonian.jpyz() * sigmaz
        
        return current
    
    def CalculateDiamagneticCurrent(self, time : float | np.ndarray[float]) -> np.ndarray[complex]:
        """Calculates the diamagnetic current.
        
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
            Has shape (2, 2, time.size), where the first and second dimensions correspond
            second and first partial derivative of H that we take respectively.
            i.e. diagmagnetic current operator j_xy(t) would be stored at index [0, 1, :].
        """
        pass
    
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