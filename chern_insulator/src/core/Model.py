import numpy as np

from data import ModelParameters, AxisData, CorrelationData, CurrentData
from . import correlation_solver, current_solver
from operators import hamiltonian

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
        self.correlationData.singleTimeFourier = correlation_solver.solve_single_time_correlations(
            self.__params
        )

        # Solves the double-time correlations using scipy ODE solver (solve_ivp).
        self.correlationData.doubleTimeCorrelations = correlation_solver.solve_double_time_correlations(
            self.__params,
            self.__axes.tAxisSec,
            self.__axes.tauAxisSec,
            self.correlationData.singleTimeFourier
        )

        self.currentData.paramagneticCurrent = current_solver.calculate_paramagnetic_current(
            self.__axes.tauAxisSec,
            self.correlationData.singleTimeFourier
        )
        
        # self.__currentData.lengthGaugeCurrent = CurrentSolver.CalculateLengthGaugeCurrent(self.__axes.tauAxisSec)

        self.currentData.diamagneticCurrent = current_solver.calculate_diamagnetic_current(
            self.__axes.tauAxisSec,
            self.correlationData.singleTimeFourier
        )
        
        self.currentData.totalCurrent = current_solver.calculate_total_current(
            hamiltonian.Ax(self.params, self.__axes.tauAxisSec),
            self.currentData.paramagneticCurrent,
            self.currentData.diamagneticCurrent
        )

        self.currentData.doubleTimeCurrent = CurrentSolver.calculate_double_time_current(
            self.__axes.tAxisSec,
            self.__axes.tauAxisSec,
            self.correlationData.singleTimeFourier,
            self.correlationData.doubleTimeCorrelations
        )

        self.currentData.meanSecondOrderCurrent = CurrentSolver.integrate_second_order_current(
            self.__params.drivingFreq,
            self.__axes.tAxisSec,
            self.currentData.doubleTimeCurrent
        )

        self.currentData.spectralNoiseTensor = CurrentSolver.calculate_spectral_noise_tensor(
            self.__params.drivingFreq,
            self.__axes.tauAxisSec,
            self.currentData.doubleTimeCurrent,
            self.__params.maxN
        )

        return self.correlationData, self.currentData
     
    @property
    def params(self) -> ModelParameters:
        return self.__params