import numpy as np

from data import ModelParameters, AxisData, CorrelationData, CurrentData
from CorrelationSolver import CorrelationSolver
from CurrentSolver import CurrentSolver
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

        self.__corrData = CorrelationData()
        self.__currentData = CurrentData()

        # Will be given in the Run() function.
        self.__axes = None

    def Run(self, axes: AxisData) -> tuple[CorrelationData, CurrentData]:
        """
        Runs all of the simulation code for this model.

        Parameters
        ----------
        axes : AxisData
            The different axes that the simulation will calculate the
            operators on.

        Returns
        -------
        CorrelationData:
            The correlation data for this model.
        CurrentData:
            The current data for this model.
        """

        # Stores the axis data.
        self.__axes = axes
        
        # Solves the single-time fourier series.
        corrSolver = CorrelationSolver(self.__params)
        self.__corrData.singleTimeFourier = corrSolver.SolveSingleTimeCorrelations()

        currentSolver = CurrentSolver(self.__params)
        self.__currentData.paramagneticCurrent = currentSolver.CalculateSingleTimeCurrent(self.__axes.tauAxisSec,
                                                                                          self.__corrData.singleTimeFourier)

        return self.__corrData, self.__currentData