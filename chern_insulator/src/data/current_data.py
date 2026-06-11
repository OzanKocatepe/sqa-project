from __future__ import annotations
from typing import Protocol
import dataclasses
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
    dc_population_variance: np.ndarray[complex] = field(default = None)
    dc_population_variance_weak_laser: np.ndarray[complex] = field(default = None)
    time_avg_generalised_noise_tensor: np.ndarray[complex] = field(default = None)
    maximal_squeezing: np.ndarray[float] = field(default = None)

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

        def add_none_safe(a, b):
            if a is not None and b is not None:
                return a + b
            return None
        
        return CurrentData(
            **{
                field.name: add_none_safe(getattr(self, field.name), getattr(other, field.name))
                for field in dataclasses.fields(self)
            }
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

        return CurrentData(
            **{
                field.name: getattr(self, field.name) / other
                if getattr(self, field.name) is not None
                else None
                for field in dataclasses.fields(self)
            }
        )
        
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
            t_averaged_second_order_current = self.t_averaged_second_order_current / other
                if self.t_averaged_second_order_current is not None
                else None,
            spectral_noise_tensor = self.spectral_noise_tensor / other
                if self.spectral_noise_tensor is not None
                else None,
            semiclassical_mode_population = self.semiclassical_mode_population / other
                if self.semiclassical_mode_population is not None
                else None,
            second_order_correlation_function = self.second_order_correlation_function / other
                if self.second_order_correlation_function is not None
                else None,
            dc_population_variance = self.dc_population_variance / other
                if self.dc_population_variance is not None
                else None
        )

    def __mul__(self, other: int | float | complex) -> CurrentData:
        """Multiplies the instance by a constant componentwise.
        
        Parameters
        ----------
        other : int | float | complex
            The integer to multiply by.
            
        Returns
        -------
        CurrentData
            A new instance with each component multiplied.
        """

        return CurrentData(
            **{
                field.name: getattr(self, field.name) * other
                if getattr(self, field.name) is not None
                else None
                for field in dataclasses.fields(self)
            }
        )