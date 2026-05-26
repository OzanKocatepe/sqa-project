from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

@dataclass(slots=True)
class CurrentData:
    """
    Stores the current data for a model.

    Parameters
    ----------
    paramagneticCurrent : ndarray[complex]
        Stores the paramagnetic current in an array of shape (2, time.size), where
        the first dimension differentiates between the x-component (index 0) and
        the y-component (index 1) of the current operator.
    diamagneticCurrent : ndarray[complex]
        Stores the xx-diamagnetic current in an array of shape (time.size,).
    totalCurrent : ndarray[complex]
        The total current, calculated from diamagnetic and paramagnetic currents,
        with shape (2, time.size).
    doubleTimeCurrent : ndarray[complex]
        Stores the double-time currents in an array of shape (2, 2, t.size, tau.size),
        with the first two axes corresponding to the left and right current direction respectively.
    lengthGaugeCurrent : ndarray[complex]
        The first-order current calculated in the length gauge.
    meanSecondOrderCurrent : ndarray[complex]
        The double-time current integrated along the t axis, with shape (2, 2, tau.size).
    spectralNoiseTensor : ndarray[complex]
        The fourier transforms at the harmonics of the driving frequency of the second order current
        (correlation tensor) at each time t.
        Of shape (2, 2, 2 * maxN + 1, t.size), where the first two axes correspond to the subscripts of
        the correlation tensor, the third axis corresponds to the harmonics, from -maxN to maxN,
        and the last axis corresponds to the time t. This is a function of t.
    """

    paramagnetic_current: np.ndarray[complex] = field(default = None)
    diamagnetic_current: np.ndarray[complex] = field(default = None)
    total_current: np.ndarray[complex] = field(default = None)
    second_order_connected_current: np.ndarray[complex] = field(default = None)
    length_gauge_total_current: np.ndarray[complex] = field(default = None)
    t_averaged_second_order_current: np.ndarray[complex] = field(default = None)
    spectral_noise_tensor: np.ndarray[complex] = field(default = None)
    semiclassical_mode_population: np.ndarray[float] = field(default = None)
    second_order_correlation_function: np.ndarray[float] = field(default = None)

    def __add__(self, other: CurrentData) -> CurrentData:
        """
        Adds together two CurrentData instances by adding together each attribute.

        Parameters
        ----------
        other : CurrentData
            The other CurrentData instance to add to this one.

        Returns
        -------
        CurrentData:
            A new CurrentData instance that has attributes equal to the sum
            of the attributes of the two operands.
        """

        return CurrentData(
            paramagnetic_current = self.paramagnetic_current + other.paramagnetic_current,
            diamagnetic_current = self.diamagnetic_current + other.diamagnetic_current,
            total_current = self.total_current + other.total_current,
            second_order_connected_current = self.second_order_connected_current + other.second_order_connected_current
                if self.second_order_connected_current is not None and other.second_order_connected_current is not None
                else None,
            length_gauge_total_current = self.length_gauge_total_current + other.length_gauge_total_current
                if self.length_gauge_total_current is not None and other.length_gauge_total_current is not None
                else None,
            t_averaged_second_order_current = self.t_averaged_second_order_current + other.t_averaged_second_order_current,
            spectral_noise_tensor = self.spectral_noise_tensor + other.spectral_noise_tensor,
            semiclassical_mode_population = self.semiclassical_mode_population + other.semiclassical_mode_population,
            second_order_correlation_function = self.second_order_correlation_function + other.second_order_correlation_function
        )
    
    def __truediv__(self, other: int) -> CurrentData:
        """Divides the instance by an int.

        We do not require other instances of division in this code currently.
        
        Parameters
        ----------
        other : int
            The integer to divide by.
            
        Returns
        -------
        CurrentData
            A new instance with each component divided by the int.
        """
        
        # No checks since we only divide in one place, and if we need to do another division
        # I trust myself to know that it passes directly to the inner arrays.
        # Could add checks later.
        return CurrentData(
            paramagnetic_current = self.paramagnetic_current / other,
            diamagnetic_current = self.diamagnetic_current / other,
            total_current = self.total_current / other,
            second_order_connected_current = self.second_order_connected_current / other
                if self.second_order_connected_current is not None
                else None,
            length_gauge_total_current = self.length_gauge_total_current / other
                if self.length_gauge_total_current is not None
                else None,
            t_averaged_second_order_current = self.t_averaged_second_order_current / other,
            spectral_noise_tensor = self.spectral_noise_tensor / other
        )