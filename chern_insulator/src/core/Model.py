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
        self.correlationData.first_order_fourier = correlation_solver.solve_single_time_correlations(
            self.__params
        )

        # Solves the double-time correlations using scipy ODE solver (solve_ivp).
        self.correlationData.second_order_correlations = correlation_solver.solve_double_time_correlations(
            self.__params,
            self.__axes.t_axis_sec,
            self.__axes.tau_axis_sec,
            self.correlationData.first_order_fourier
        )

        self.currentData.paramagnetic_current = current_solver.calculate_paramagnetic_current(
            self.__params,
            self.__axes.tau_axis_sec,
            self.correlationData.first_order_fourier
        )
        
        # self.__currentData.lengthGaugeCurrent = CurrentSolver.CalculateLengthGaugeCurrent(self.__axes.tauAxisSec)

        self.currentData.diamagnetic_current = current_solver.calculate_diamagnetic_current(
            self.__params,
            self.__axes.tau_axis_sec,
            self.correlationData.first_order_fourier
        )
        
        self.currentData.total_current = current_solver.calculate_total_current(
            hamiltonian.Ax(self.params, self.__axes.tau_axis_sec),
            self.currentData.paramagnetic_current,
            self.currentData.diamagnetic_current
        )

        self.currentData.second_order_connected_current = current_solver.calculate_double_time_current(
            self.__params,
            self.__axes.t_axis_sec,
            self.__axes.tau_axis_sec,
            self.correlationData.first_order_fourier,
            self.correlationData.second_order_correlations
        )

        self.currentData.t_averaged_second_order_current = current_solver.integrate_second_order_current(
            self.__params.drivingFreq,
            self.__axes.t_axis_sec,
            self.currentData.second_order_connected_current
        )

        self.currentData.spectral_noise_tensor = current_solver.calculate_spectral_noise_tensor(
            self.__params.drivingFreq,
            self.__axes.tau_axis_sec,
            self.currentData.second_order_connected_current,
            self.__params.maxN
        )

        current_fourier_coefficients = current_solver.calculate_current_fourier_coefficients(
            self.__params,
            self.currentData.total_current,
            self.__axes.tau_axis_sec,
            self.__params.maxN
        )

        semiclassical_intracavity_field_amplitude = current_solver.calculate_semiclassical_intracavity_field_amplitude(
            self.__params,
            current_fourier_coefficients,
            self.__axes.tau_axis_sec,
            self.__params.maxN
        )

        self.currentData.semiclassical_mode_population = current_solver.calculate_semiclassical_mode_population(
            self.__params,
            self.__axes.tau_axis_sec,
            semiclassical_intracavity_field_amplitude
        )

        self.currentData.second_order_correlation_function = current_solver.calculate_second_order_correlation_function(
            self.__params,
            self.__axes.tau_axis_sec,
            semiclassical_intracavity_field_amplitude,
            self.currentData.semiclassical_mode_population
        )

        return self.correlationData, self.currentData
     
    @property
    def params(self) -> ModelParameters:
        return self.__params