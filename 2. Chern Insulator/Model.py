import numpy as np
from scipy import integrate

from data import ModelParameters
from CorrelationSolver import CorrelationSolver
from Hamiltonian import Hamiltonian

class Model:
    """Contains a Chern Insulator model evaluated at a single pair kx, ky."""

    def __init__(self, params: ModelParameters) -> None:
        """Initialises the instance.
        
        Parameters
        ----------
        params : ModelParameters
            The parameters to use for this model.
        """

        self.__params = params
        self.__hamiltonian = Hamiltonian(self.__params)

    def Run(self, tauMax: float):
        """
        Runs all of the simulation code for this model.

        Parameters
        ----------
        tauMax : float
            The maximum non-dimensional time for the system run for.
        """
        
        # Solves the single-time fourier series.

        tauAxisDim = np.linspace(0, tauMax, 4000)
        tauAxisSec = tauAxisDim / self.__params.decayConstant

        corrSolver = CorrelationSolver(self.__params)
        singleTimeFourier = corrSolver.SolveSingleTimeCorrelations()

        # Solves the single-time ODE.
        singleTime = integrate.solve_ivp(
            fun = self.__hamiltonian.EquationsOfMotion,
            t_span = (0, tauMax / self.__params.decayConstant),
            y0 = np.array([0, 0, -1], dtype=complex),
            t_eval = tauAxisSec,
            rtol = 1e-9,
            atol = 1e-12,
            vectorized = True
        ).y

        return singleTimeFourier, singleTime