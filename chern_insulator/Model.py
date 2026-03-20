import numpy as np

from data import ModelParameters, AxisData, CorrelationData, CurrentData
from CorrelationSolver import CorrelationSolver
from CurrentSolver import CurrentSolver
from Hamiltonian import Hamiltonian

class Model:
    """Contains a Chern Insulator model evaluated at a single pair kx, ky."""

    def __init__(self, params: ModelParameters, axes: AxisData) -> None:
        """Initialises the instance.
        
        Parameters
        ----------
        params : ModelParameters
            The parameters to use for this model.
        """

        self.__params = params
        self.__axes = axes
        self.__hamiltonian = Hamiltonian(self.__params)

        self.__corrData = CorrelationData()
        self.__currentData = CurrentData()

    def Run(self) -> tuple[CorrelationData, CurrentData]:
        """
        Runs all of the simulation code for this model.

        Returns
        -------
        CorrelationData:
            The correlation data for this model.
        CurrentData:
            The current data for this model.
        """
        
        # Solves the single-time fourier series.
        corrSolver = CorrelationSolver(self.__params)
        self.__corrData.singleTimeFourier = corrSolver.SolveSingleTimeCorrelations()

        currentSolver = CurrentSolver(self.__params)
        self.__currentData.paramagneticCurrent = currentSolver.CalculateSingleTimeCurrent(self.__axes.tauAxisSec,
                                                                                          self.__corrData.singleTimeFourier)

        return self.__corrData, self.__currentData