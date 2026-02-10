from dataclasses import dataclass
import numpy as np
from .Fourier import Fourier
from .SSHParameters import ModelParameters
from .CorrelationData import CorrelationData
import scipy.special as special

@dataclass
class CurrentData:
    """Contains all of the data relating to the current operator."""

    # The expectation of the current operator in the time- and frequency- domains.
    timeDomainCurrent: np.ndarray[complex] = None
    freqDomainCurrent: np.ndarray[complex] = None

    # The fourier series of the time-domain current.
    currentFourierSeries: Fourier = None

    # The expectation of the double-time current operator.
    doubleTimeCurrent: np.ndarray[complex] = None
    # The expectation of the double-time current operator, integrated w.r.t. t.
    integratedDoubleTimeCurrent: np.ndarray[complex] = None
    # The product of the expectation of the current operator, evaluated at two different times.
    doubleTimeCurrentProduct: np.ndarray[complex] = None

    # The connected correlator for the double-time current operator in time- and frequency- domains.
    timeConnectedCorrelator: np.ndarray[complex] = None
    freqConnectedCorrelator: np.ndarray[complex] = None

    # The fourier transform of the connected-correlator evaluated at integer harmonics of the
    # driving frequency.
    harmonics : np.ndarray[complex] = None

    # The double-time current product (second half of the connected correlator) evaluated
    # numerically using the timeDomainCurrent at multiple times, rather than using the fourier series
    # derived analytically. Used to make the product term in the connected correlator is correct.
    _numericalDoubleTimeCurrentProduct: np.ndarray[float] = None

    def __add__(self, other: CurrentData) -> CurrentData:
        """
        Adds another CurrentData instance to itself to create a new CurrentData instance.

        Parameters
        ----------
        other : CurrentData
            The other CurrentData object to add to this one.

        Returns
        -------
        CurrentData
            The new current data instance with the sum of their time and frequency data.
        """

        return CurrentData(
            timeDomainCurrent = self.timeDomainCurrent + other.timeDomainCurrent,
            freqDomainCurrent = self.freqDomainCurrent + other.freqDomainCurrent,
            currentFourierSeries = self.currentFourierSeries + other.currentFourierSeries,
            doubleTimeCurrent = self.doubleTimeCurrent + other.doubleTimeCurrent,
            integratedDoubleTimeCurrent = self.integratedDoubleTimeCurrent + other.integratedDoubleTimeCurrent,
            doubleTimeCurrentProduct = self.doubleTimeCurrentProduct + other.doubleTimeCurrentProduct,
            timeConnectedCorrelator = self.timeConnectedCorrelator + other.timeConnectedCorrelator,
            freqConnectedCorrelator = self.freqConnectedCorrelator + other.freqConnectedCorrelator,
            harmonics = self.harmonics + other.harmonics,
            _numericalDoubleTimeCurrentProduct = self._numericalDoubleTimeCurrentProduct + other._numericalDoubleTimeCurrentProduct
        )

    def CalculateFourier(self, k: float, params: ModelParameters, correlationData: CorrelationData) -> None:
        """
        Calculates the fourier expansions of the coefficients of the pauli expectations, when the current expectation is written in
        terms of the pauli expectations, and the fourier expansions of the current expectation on its own.
        
        Parameters
        ----------
        k : float
            The momentum of the system.
        params : ModelParameters
            The parameters of the SSH model.
        correlationData : CorrelationData
            The correlation data calculated within the relevant SSH instance.
        """

        # Calculates value of current coefficients and stores them in coefficientFourierExpansion.
        coefficientFourierExpansions = self.__CalculateCurrentCoefficients(k, params, correlationData.tauAxisSec)
        # Calculates the full fourier expansion of the current expectation and stores it in currentFourierSeries.
        self.__CalculateCurrentExpectationCoefficients(params, correlationData, coefficientFourierExpansions)

    def __CalculateCurrentExpectationCoefficients(self, params: ModelParameters, correlationData: CorrelationData, coefficientFourierExpansions: list[Fourier]) -> None:
        r"""
        Calculates the coefficients for the fourier expansion of the expectation of the current operator
        into the laser harmonics. These are calculated from the fourier coefficients of the single-time
        correlations $\langle \sigma_-(t) \rangle,\, \langle \sigma_+(t) \rangle,\, \langle \sigma_z(t) \rangle$, and the current coefficient functions $j_-(t), j_+(t), j_z(t)$.

        Parameters
        ----------
        params : ModelParameters
            The parameters of the SSH model.
        correlationData : CorrelationData
            The correlation data calculated within the relevant SSH instance.
        coefficientFourierExpansions : list[Fourier]
            A list containing the fourier series for the coefficients of the pauli matrices when the current operator is expanded
            into the pauli matrix basis.
        """

        # The value of n must be the same as that of the expectation and current coefficient fourier expansions,
        # so we just grab it from the first expectation expansion.
        n = correlationData.singleTimeFourier[0].n

        coefficients = np.zeros((4 * n + 1), dtype=complex)
        expectationCoeff = np.zeros((3, 2 * n + 1), dtype=complex)
        currentCoeff = np.zeros((3, 2 * n + 1), dtype=complex)
        for i in range(3):
            expectationCoeff[i] = correlationData.singleTimeFourier[i].coeffs
            currentCoeff[i] = coefficientFourierExpansions[i].coeffs

        r"""
        The following code is slightly convoluted, so a better explanation is given here. We have two fourier expansions.
        
        $\sum_{n = -N}^N a_n e^{in \omega t}, \quad \sum_{m = -N}^N b_m e^{im \omega t}$

        We want to multiply these together. Clearly, this is just

        $\sum_{n = -N}^N \sum_{m = -N}^N a_n b_m e^{i(n + m) \omega t}$

        However, we want this as a single sum in the basis of $e^{in\omega t}$ so that we can express the entire thing as a
        fourier expansion. To do this, we will consider that the only possible values of $n + m$ are $[-2N, 2N] \subseteq \mathbb{Z}$.
        We also know that for any fixed $c := n + m$, the two coefficients that form a coefficient of $e^{ic\omega t}$ are $a_c b_{c - n}$ for every possible value of $n$ that we can achieve. Hence, the final form of the system is

        $\sum_{n = -2N}^{2N} \left( \sum_{m} a_m b_{n - m} \right) e^{in \omega t}$

        where we sum over all the possible values of $m$ that allow both coefficients to exist. So, for each fixed $n$, we will sum over all the $m$ such that $m \in [-N, N]$ and $n - m \in [-N, N]$. Hence, in code, we can loop over all the values of $m \in [-N, N]$ and manually check if $n - m \in [-N, N]$ for the current values of $m$ and $n$, and only add that term to the
        current $n$th coefficient of the final fourier expansion if we return true.

        The code below results in a current fourier expansion in terms of the laser harmonics that matches the current for a fixed $k$
        exactly. We can also just sum the coefficients over multiple $k$ to find the total current over multiple $k$.
        """

        # Loops over all possible integer harmonics of the laser frequency that can be produced
        # by multiplying the two fourier transforms together, from -2n to 2n.
        for frequencySum in range(-2 * n, 2 * n + 1):
            # Loops through every possible coefficient index, from -n to n, of the first coefficient term.
            for firstIndex in range(-n, n + 1):
                # Checks if the required second coefficient index required to make the two indices sum to frequencySum exists.
                # Only adds it if it does.
                if np.abs(frequencySum - firstIndex) <= n:
                    for functionIndex in range(3):
                        coefficients[frequencySum + 2 * n] += currentCoeff[functionIndex, firstIndex + n] * expectationCoeff[functionIndex, frequencySum - firstIndex + n]

        self.currentFourierSeries = Fourier(
            params.drivingFreq,
            coefficients
        )

    def __CalculateCurrentCoefficients(self, k: float, params: ModelParameters, tauAxisSec: np.ndarray[float]) -> list[Fourier]:
        r"""
        Calculates the fourier coefficients corresponding to the coefficients of the operators in the current operator.
        i.e. the coefficients for $j_-(t), j_+(t), j_z(t)$.

        Parameters
        ----------
        k : float
            The momentum of the system.
        params : ModelParameters
            The parameters of the SSH model.
        tauAxisSec : ndarray[float]
            The points in time, in seconds, that the current operator in the time-domain was calculated at.

        Returns
        -------
        list[Fourier]
            A list containing the fourier series for each of the coefficient functions to the pauli matrices when the current
            operator is expanded in terms of the pauli matrices.
        """

        n = Fourier.DetermineMaxN(tauAxisSec, params.drivingFreq)
        coefficients = np.zeros((3, 2 * n + 1), dtype=complex)
 
        theta = k - params.phiK

        for i in np.arange(-n, n + 1):
            # Coefficient for $j_-(t)$.
            coefficients[0, i + n] = -0.5j * params.t2 * special.jv(i, params.drivingAmplitude) * (float(-1)**i * np.exp(1j * theta) + np.exp(-1j * theta))
            
            # Coefficient for $j_z(t)$.
            coefficients[2, i + n] = 0.5j * params.t2 * special.jv(i, params.drivingAmplitude) * (float(-1)**i * np.exp(1j * theta) - np.exp(-1j * theta))

        # Using $j_+(t) = -j_-(t)$, we can calculate the remaining coefficients.
        coefficients[1, :] = -coefficients[0, :]


        coefficientFourierExpansion = []
        for i in range(3):
            coefficientFourierExpansion.append(
                Fourier(
                    params.drivingFreq,
                    coefficients[i, :]
                )
            )

        return coefficientFourierExpansion