import numpy as np

from data import ModelParameters, AxisData, ModelData, CorrelationData
from solvers import correlation_solver, model_solver
from physics import hamiltonian

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

        self.model_data = ModelData()
        self.correlation_data = CorrelationData()

        # Will be given in the Run() function.
        self.__axes = None

    def Run(self, axes: AxisData, disable_second_order: bool) -> tuple[ModelData]:
        """
        Runs all of the simulation code for this model.

        Parameters
        ----------
        axes : AxisData
            The different axes that the simulation will calculate the
            operators on.
        disable_second_order : bool
            Whether to disable the second-order calculations.

        Returns
        -------
        modelData:
            The correlation data for this model.
        modelData:
            The current data for this model.
        """

        # Stores the axis data.
        self.__axes = axes

        # SINGLE-TIME PROPERTIES
        # ----------------------
        
        # Solves the single-time fourier series.
        self.correlation_data.first_order_fourier = correlation_solver.solve_single_time_correlations(
            self.__params
        )

        self.model_data.paramagnetic_current = model_solver.calculate_paramagnetic_current(
            self.__params,
            self.__axes.tau_axis_sec,
            self.correlation_data.first_order_fourier
        )
        
        self.model_data.diamagnetic_current = model_solver.calculate_diamagnetic_current(
            self.__params,
            self.__axes.tau_axis_sec,
            self.correlation_data.first_order_fourier
        )
        
        self.model_data.total_current = model_solver.calculate_total_current(
            hamiltonian.Ax(self.params, self.__axes.tau_axis_sec),
            self.model_data.paramagnetic_current,
            self.model_data.diamagnetic_current
        )

        # Defines consistent scattering rates.
        scattering_rates = np.logspace(-4, -1, 5) * hamiltonian.find_band_gap(self.__params.delta, resolution=100)

        self.model_data.dc_population_variance_weak_laser, noise_correlation_tensor_weak_laser_real = model_solver.calculate_dc_population_variance_weak_laser_power(
            self.__params,
            scattering_rates
        )

        self.model_data.time_avg_generalised_noise_tensor_weak_laser = model_solver.calculate_weak_laser_noise_tensor(
            self.__params,
            scattering_rates,
            noise_correlation_tensor_weak_laser_real
        )

        # DOUBLE-TIME PROPERTIES
        # ----------------------

        if not disable_second_order:
            # Solves the double-time correlations using scipy ODE solver (solve_ivp).
            self.correlation_data.second_order_correlations = correlation_solver.solve_double_time_correlations(
                self.__params,
                self.__axes.t_axis_sec,
                self.__axes.tau_axis_sec,
                self.correlation_data.first_order_fourier
            )

            self.model_data.second_order_current, self.model_data.second_order_current_product, self.model_data.matter_correlation_tensor = model_solver.calculate_double_time_current(
                self.__params,
                self.__axes.t_axis_sec,
                self.__axes.tau_axis_sec,
                self.correlation_data.first_order_fourier,
                self.correlation_data.second_order_correlations
            )

        return self.model_data
     
    @property
    def params(self) -> ModelParameters:
        return self.__params