from dataclasses import dataclass, field
import numpy as np

@dataclass(slots=True)
class EnsembleData:
    """Stores the data that will be calculated for the entire ensemble at once (i.e. those which are calculated after
    BZ-averaging)."""

    # These are the first-order products.
    semiclassical_mode_population: np.ndarray[complex] = None
    second_order_correlation_function: np.ndarray[complex] = None
    squeezing_weak_laser: np.ndarray[complex] = None
    angular_momentum: np.ndarray[complex] = None

    # These are the second-order products, i.e. those that rely on the second-order correlations
    # or the second order connected current.
    time_avg_second_order_connected_current: np.ndarray[complex] = None
    spectral_noise_tensor: np.ndarray[complex] = None
    dc_population_variance: np.ndarray[complex] = None
    generalised_noise_tensor: np.ndarray[complex] = None
    squeezing: np.ndarray[complex] = None