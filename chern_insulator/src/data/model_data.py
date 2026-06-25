from dataclasses import dataclass, fields
import numpy as np

from . import Fourier

@dataclass(slots=True)
class ModelData:
    """Stores the data that will be calculated within each model (i.e. at each k-point). This will then be averaged
    over the Brillouin Zone."""

    # These are the first-order products calculated at each momentum.
    paramagnetic_current: np.ndarray[complex] = None
    diamagnetic_current: np.ndarray[complex] = None
    total_current: np.ndarray[complex] = None

    # These are the weak_laser approximations, also first order.
    dc_population_variance_weak_laser: np.ndarray[complex] = None
    time_avg_generalised_noise_tensor_weak_laser: np.ndarray[complex] = None

    # These are the second-order products which must be calculated per-momentum.
    second_order_connected_current: np.ndarray[complex] = None

    def __add__(self, other: ModelData) -> ModelData:
        def add_none_safe(a, b):
            if a is not None and b is not None:
                return a + b
            return None
        
        return ModelData(
            **{
                field.name: add_none_safe(getattr(self, field.name), getattr(other, field.name))
                for field in fields(self)
            }
        )
 
    def __truediv__(self, other: int) -> ModelData:
        return ModelData(
            **{
                field.name: getattr(self, field.name) / other
                if getattr(self, field.name) is not None
                else None
                for field in fields(self)
            }
        )
        
    def __mul__(self, other: int | float | complex) -> ModelData:
        return ModelData(
            **{
                field.name: getattr(self, field.name) * other
                if getattr(self, field.name) is not None
                else None
                for field in fields(self)
            }
        )