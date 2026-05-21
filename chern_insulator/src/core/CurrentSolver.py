import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from typing import Callable

from data import ModelParameters, Fourier
from operators import Hamiltonian, ParamagneticCurrentX, ParamagneticCurrentY, DiamagneticCurrentX
from LengthGauge import LengthGauge

class CurrentSolver:
    """Contains the code for solving for single- and double- time currents."""

    def __init__(self, params: ModelParameters, hamiltonian: Hamiltonian) -> None:
        """
        Initialises the current solver.

        Parameters
        ----------
        params : ModelParameters
            The parameters of the model we are solving the correlations for.
        hamiltonian: Hamiltonian
            The hamiltonian of the system.
        """

        self.__params = params
        self.__hamiltonian = hamiltonian
        self.__jpx = ParamagneticCurrentX(self.__params, self.__hamiltonian)
        self.__jpy = ParamagneticCurrentY(self.__params, self.__hamiltonian)
        self.__jdx = DiamagneticCurrentX(self.__params, self.__hamiltonian)

    def CalculateParamagneticCurrent(self, time: float | np.ndarray[float], fourierSeries: list[Fourier]) -> np.ndarray[complex]:
        """Calculates the paramagnetic current.
        
        Parameters
        ----------
        time : float | ndarray[float]
            The points in time, in seconds, to evaluate the paramagnetic current operator at.
        fourierSeries : list[Fourier]
            The list containing the Fourier series for sigma_-, sigma_+,
            and sigma_z, in that order.

        Returns
        -------
        ndarray[complex]:
            The value of the paramagnetic current operator at the corresponding times.
            Has shape (2, time.size), where the first dimension corresponds to the
            current in the x-dimension and y-dimension for indices 0 and 1 respectively.
        """

        current = np.zeros((2, time.size), dtype=complex)

        sigmam = fourierSeries[0].Evaluate(time)
        sigmap = fourierSeries[1].Evaluate(time)
        sigmaz = fourierSeries[2].Evaluate(time)
 
        current[0, :] = (
            self.__jpx.minus(time) * sigmam
            + self.__jpx.plus(time) * sigmap
            + self.__jpx.z(time) * sigmaz
        )
 
        current[1, :] = (
            self.__jpy.minus(time) * sigmam
            + self.__jpy.plus(time) * sigmap
            + self.__jpy.z(time) * sigmaz
        )
        
        return current
    
    def CalculateDiamagneticCurrent(self, time : float | np.ndarray[float], fourierSeries: list[Fourier]) -> np.ndarray[complex]:
        """Calculates the xx-diamagnetic current.
        
        Parameters
        ----------
        time : float | ndarray[float]
            The points in time, in seconds, to evaluate the diamagnetic current operator at.
        fourierSeries : list[Fourier]
            The list containing the Fourier series for sigma_-, sigma_+,
            and sigma_z, in that order.

        Returns
        -------
        ndarray[complex]:
            The value of the diamagnetic current operator at the corresponding times.
            Has shape (time.size,), since we only need the xx-component of the diamagnetic
            current in our case.
        """

        sigmam = fourierSeries[0].Evaluate(time)
        sigmap = fourierSeries[1].Evaluate(time)
        sigmaz = fourierSeries[2].Evaluate(time)
        
        return (
            self.__jdx.minus(time) * sigmam
            + self.__jdx.plus(time) * sigmap
            + self.__jdx.z(time) * sigmaz
        )
    
    def CalculateTotalCurrent(self,
        Ax: np.ndarray[float],
        paramagnetic: np.ndarray[complex],
        diamagnetic: np.ndarray[complex]
    ) -> np.ndarray[complex]:
        """Calculates the total current given the paramagnetic and diamagnetic current.
        
        Parameters
        ----------
        Ax : ndarray[float], shape (n,)
            The driving vector potential calculated at each time in the tauAxis, such that
            each index here corresponds to the vector potential that was applied at the same index
            (corresponding to the same time) in the paramagentic and diamagnetic currents.
        paramagnetic : ndarray[complex], shape(2, n)
            The paramagentic current array, as stored in CurrentData and calculated in
            CalculateParamagneticCurrent.
        diamagnetic : ndarray[complex], shape(n,)
            The diamagnetic current array, as stored in CurrentData and calculated in
            CalculateDiamagneticCurrent.

        Returns
        -------
        ndarray[complex]:
            An array of shape (2, n) containing the total current in each direction. In the x-direcition
            this is a combination of the paramagnetic and diamagnetic currents. In the y-direction, this is
            just the paramagnetic current.
        """

        totalCurrent = np.zeros(paramagnetic.shape, dtype=complex)

        # Total x-current depends on paramagnetic and diamagnetic x-currents, along with vector potential.
        totalCurrent[0, :] = paramagnetic[0, :] + diamagnetic * Ax
        # Total y-current remains just the paramagnetic y-current.
        totalCurrent[1, :] = paramagnetic[1, :]

        return totalCurrent
    
    def CalculateDoubleTimeCurrent(self,
        tAxis: np.ndarray[float],
        tauAxis: np.ndarray[float],
        singleTimeFourier: list[Fourier],
        doubleTimeCorrelators: np.ndarrayp[complex]
    ) -> np.ndarray[complex]:
        """Calculates the double-time current correlators.

        Parameters
        ----------
        tAxis : ndarray[float], shape (n,)
            The tAxis, as stored in AxisData, in seconds.
        tauAxis : ndarray[float], shape(n,)
            The tauAxis, as stored in AxisData, in seconds.
        singleTimeFourier : list[Fourier]
            The single-time correlation Fourier series.
        doubleTimeCorrelators : ndarray[complex], shape(3, 3, tAxis.size, tauAxis.size)
            The double-time correlators.
        
        Returns
        -------
        ndarray[complex]
            An array of shape (2, 2, tAxis.size, tauAxis.size) containing the double-time current correlations.
            The first and second axes correspond to the direction of the current operator, with indices
            0 and 1 corresponding to the x- and y- current respectively.
            The last two axes correspond to the double-time correlator j_alpha(t) j_beta(t + tau) at times
            t and t + tau.
        """

        doubleCurrentCorrelations = np.zeros((2, 2, tAxis.size, tauAxis.size), dtype=complex)
        tPlusTauAxis = np.add.outer(tAxis, tauAxis)

        currentOperators = [self.__jpx, self.__jpy]
        for leftDirection in range(2):
            for rightDirection in range(2):

                # Now that we know the direction of our operators, we can calculate
                # the coefficients individually.
                firstOperatorCoefficients = np.array([
                    currentOperators[leftDirection].minus(tAxis),
                    currentOperators[leftDirection].plus(tAxis),
                    currentOperators[leftDirection].z(tAxis),
                ])

                secondOperatorCoefficients = np.array([
                    currentOperators[rightDirection].minus(tPlusTauAxis.flatten()).reshape(tPlusTauAxis.shape),
                    currentOperators[rightDirection].plus(tPlusTauAxis.flatten()).reshape(tPlusTauAxis.shape),
                    currentOperators[rightDirection].z(tPlusTauAxis.flatten()).reshape(tPlusTauAxis.shape),
                ])

                # Now we can loop through each of the possible combinations of these coefficients,
                # and multiply by the respective connected correlator.
                for firstCoeff in range(3):
                    for secondCoeff in range(3):
                        # Calculates the product term in the connected correlator.
                        prod = (singleTimeFourier[firstCoeff].Evaluate(tAxis)[:, np.newaxis]
                                * singleTimeFourier[secondCoeff].Evaluate(tPlusTauAxis))
                        # Calculates the connected correlator.
                        connected = doubleTimeCorrelators[firstCoeff, secondCoeff, :, :] - prod
                        # Multiplies the first and second coefficient, resulting in an array of shape
                        # (t.size, tau.size). Then, multiplies that by the connected correlator at each point.
                        # Adds this onto the connected current correlator for this direction, so by the end
                        # of the loop we've added on every contribution for this direction.
                        doubleCurrentCorrelations[leftDirection, rightDirection, :, :] += (
                            firstOperatorCoefficients[firstCoeff][:, np.newaxis]
                            * secondOperatorCoefficients[secondCoeff]
                            * connected
                        )

        return doubleCurrentCorrelations 
  
    def CalculateLengthGaugeCurrent(self, time: float | np.ndarray[float]) -> np.ndarray[complex]:
        """Calculates the expectation value of the current in the length gauge.
        
        Parameters
        ----------
        time : float | ndarray[float]
            The points in time, in seconds, to evaluate the current operator at.

     
        Returns
        -------
        ndarray[complex]:
            The value of the length gauge current operator at the corresponding times.
            Has shape (2, time.size), where the first dimension corresponds to the
            current in the x-dimension and y-dimension for indices 0 and 1 respectively.
        """

        current = np.zeros((2, time.size), dtype=complex)

        lg = LengthGauge(self.__params, self.__hamiltonian)

        # Solves the ODE for our density matrix at the desired times.
        rho = integrate.solve_ivp(
            fun = lg.DensityMatrixODE,
            t_span = (0, np.max(time)),
            y0 = np.array([0.0, 0.0, 0.0, 1.0], dtype=complex),
            t_eval = time,
            rtol=1e-9,
            atol=1e-12
        ).y.T

        # Reshapes rho into a matrix rather than a flattened array.
        rho = rho.reshape((time.size, 2, 2))

        # Calculates our current operators at each desired time.
        jx = lg.jxLengthGauge(time)
        jy = lg.jyLengthGauge(time)

        # Calculates the average current.
        current[0, :] = np.trace(jx @ rho, axis1=1, axis2=2)
        current[1, :] = np.trace(jy @ rho, axis1=1, axis2=2)

        return current
    
    def IntegrateSecondOrderCurrent(self, drivingFreq: float, tAxis: np.ndarray[float], doubleTimeCurrent: np.ndarray[complex]) -> np.ndarray[complex]:
        """Integrates the second-order current functions over the t-axis.

        Parameters
        ----------
        drivingFreq: float
            The driving frequency, in Hz, of the pumping, which is also the frequency of the system
            in steady state.
        tAxis : ndarray[float]
            The tAxis in seconds.
        doubleTimeCurrent : ndarray[complex]
            The second-order current values, of shape (2, 2, t.size, tau.size), where
            the first two axes are the left and right current direction.

        Returns
        -------
        ndarray[complex]
            The doubleTimeCurrent, integrated over the t-axis (axis 2) and divided by the driving period,
            in order to get the mean current at each tau, resulting in an array of shape (2, 2, tau.size).
        """

        return (1 / drivingFreq) * np.trapezoid(
            y = doubleTimeCurrent,
            x = tAxis,
            axis = 2
        )