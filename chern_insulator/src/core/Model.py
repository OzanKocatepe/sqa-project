import numpy as np

from data import ModelParameters, AxisData, CorrelationData, CurrentData
from .CorrelationSolver import CorrelationSolver
from .CurrentSolver import CurrentSolver
from operators import Hamiltonian

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

        self.correlationData = CorrelationData()
        self.currentData = CurrentData()

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
        corrSolver = CorrelationSolver(self.__params, self.__hamiltonian)
        self.correlationData.singleTimeFourier = corrSolver.SolveSingleTimeCorrelations()

        # Solves the double-time correlations using scipy ODE solver (solve_ivp).
        self.correlationData.doubleTimeCorrelations = corrSolver.SolveDoubleTimeCorrelations(
            self.__axes.tAxisSec,
            self.__axes.tauAxisSec,
            self.correlationData.singleTimeFourier
        )

        currentSolver = CurrentSolver(self.__params, self.__hamiltonian)
        self.currentData.paramagneticCurrent = currentSolver.CalculateParamagneticCurrent(
            self.__axes.tauAxisSec,
            self.correlationData.singleTimeFourier
        )
        
        # self.__currentData.lengthGaugeCurrent = currentSolver.CalculateLengthGaugeCurrent(self.__axes.tauAxisSec)

        self.currentData.diamagneticCurrent = currentSolver.CalculateDiamagneticCurrent(
            self.__axes.tauAxisSec,
            self.correlationData.singleTimeFourier
        )
        
        self.currentData.totalCurrent = currentSolver.CalculateTotalCurrent(
            self.__hamiltonian.Ax(self.__axes.tauAxisSec),
            self.currentData.paramagneticCurrent,
            self.currentData.diamagneticCurrent
        )

        self.currentData.doubleTimeCurrent = currentSolver.CalculateDoubleTimeCurrent(
            self.__axes.tAxisSec,
            self.__axes.tauAxisSec,
            self.correlationData.singleTimeFourier,
            self.correlationData.doubleTimeCorrelations
        )

        return self.correlationData, self.currentData
    
    # @property
    # def currentData(self) -> CurrentData:
    #     return self.__currentData
    
    # @property
    # def correlationData(self) -> CorrelationData:
    #     return self.__corrData
    
    @property
    def params(self) -> ModelParameters:
        return self.__params