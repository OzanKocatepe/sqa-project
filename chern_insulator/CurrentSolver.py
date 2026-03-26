import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

from data import ModelParameters, Fourier
from Hamiltonian import Hamiltonian

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

    def CalculateSingleTimeCurrent(self, time: float | np.ndarray[float], fourierSeries: list[Fourier]) -> np.ndarray[complex]:
        """Calculates the single time current.
        
        Parameters
        ----------
        time : float | ndarray[float]
            The points in time, in seconds, to evaluate the current operator at.
        fourierSeries : list[Fourier]
            The list containing the Fourier series for sigma_-, sigma_+,
            and sigma_z, in that order.

        Returns
        -------
        ndarray[complex]:
            The value of the current operator at the corresponding times.
            Has shape (2, time.size), where the first dimension corresponds to the
            current in the x-dimension and y-dimension for indices 0 and 1 respectively.
        """

        current = np.zeros((2, time.size), dtype=complex)

        sigma = self.__SolveSigmaNumerically(time)
        sigmam = sigma[0, :]
        sigmap = sigma[1, :]
        sigmaz = sigma[2, :]

        # sigmam = fourierSeries[0].Evaluate(time)
        # sigmap = fourierSeries[1].Evaluate(time)
        # sigmaz = fourierSeries[2].Evaluate(time)

        # plt.plot(time, sigmam - np.conjugate(sigmap))
        # plt.title(r"$\sigma_- - \sigma_+^*$")
        # plt.show()
        # print(np.mean(sigmam - np.conjugate(sigmap)))
        
        current[0, :] = self.__hamiltonian.jxm(time) * sigmam \
            + self.__hamiltonian.jxp(time) * sigmap \
            + self.__hamiltonian.jxz(time) * sigmaz
        
        current[1, :] = self.__hamiltonian.jym() * sigmam \
            + self.__hamiltonian.jyp() * sigmap \
            + self.__hamiltonian.jyz() * sigmaz
        
        return current
    
    def __SolveSigmaNumerically(self, time: np.ndarray[float]) -> np.ndarray[complex]:
        """
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

        return integrate.solve_ivp(
            fun = self.__hamiltonian.EquationsOfMotion,
            t_span = (0, np.max(time)),
            y0 = np.array([0.0, 0.0, -1.0], dtype=complex),
            t_eval = time,
            rtol=1e-9,
            atol=1e-12,
            vectorized = True
        ).y