import numpy as np
import scipy.integrate as integrate
import scipy.special as special

from .data import *

class SSH:
    """
    Contains the core physics and calculations for the 1-dimensional SSH model.
    Since we consider a fixed momentum, this system simplifies to essentially a two-level system.
    """

    def __init__(self, params: ModelParameters):
        """
        Constructs an instance of the SSH model.

        Parameters
        ----------
        params: ModelParameters
            The parameters of the SSH model.
        """

        self.__params = params

        # The data that hasn't been created yet.
        self.__correlationData: CorrelationData = CorrelationData()
        self.__currentData: CurrentData = CurrentData()
        self.__axes: AxisData = None
        self.__diagnosticData = DiagnosticData()

    def __SinusoidalDrivingTerm(self, t: float | np.ndarray[float]) -> float | np.ndarray[float]:
        """A classical, sinusoidal driving term for the system.
        
        Parameters
        ----------
        t : float | ndarray[float]
            The points in time (in seconds) to evaluate the driving term at.
            
        Returns
        -------
        float | ndarray[float]
            The value of the driving term at each time t.
        """

        return self.__params.drivingAmplitude * np.sin(2 * np.pi * self.__params.drivingFreq * t)

    def __ClassicallyDrivenSSHEquations(self, t: float, c: np.ndarray[complex], inhomPart: float) -> np.ndarray[float]:
        r"""
        The ODE (equations of motion) for the single-time expectations of $\sigma_-(t)$, $\sigma_+(t)$, and $\sigma_z(t)$. 

        Parameters
        ----------
        t : float
            The time t, in seconds.
        c : ndarray[float]
            The 3-component single-time correlation array.
        inhomPart : float
            The inhomogenous part of the system, which changes depending on the
            order of the correlation function. The correct inhomogenous parts are determined
            outside of this function.

        Returns:
        --------
        ndarray[float] 
            The value of dc/dt at some time t.
        """

        # Defines the useful parameters.
        A = self.__SinusoidalDrivingTerm
        vZ = 2 * self.__params.t2 * np.sin(self.__params.k - self.__params.phiK - 0.5 * A(t)) * np.sin(0.5 * A(t))
        vPm = 2j * self.__params.t2 * np.cos(self.__params.k - self.__params.phiK - 0.5 * A(t)) * np.sin(0.5 * A(t))

        # Defines the coefficient matrix.
        B = np.array([
            [
                -2j * (np.abs(self.__params.Ek) + vZ) - 0.5 * self.__params.decayConstant,
                0,
                -1j * vPm
            ],
            [
                0,
                2j * (np.abs(self.__params.Ek) + vZ) - 0.5 * self.__params.decayConstant,
                -1j * vPm
            ],
            [
                2j * vPm,
                2j * vPm,
                -self.__params.decayConstant
            ]], dtype=complex)
    
        # Inhomogenous part.
        d = np.array([0, 0, inhomPart], dtype=complex)
    
        return B @ c + d
    
    def SolveCorrelations(self, axes: AxisData, initialConditions: np.ndarray[complex]) -> None:
        r"""
        Solves the system of ODEs for the expectations of our two-level system operators $(\langle \sigma_i(t) \rangle$,
        and the double-time correlation functions $\langle \sigma_i (t) \sigma_j (t + \tau) \rangle$, for $i, j \in \{ -, +, z \}$.
            
        Parameters
        ----------
        axes : AxisData
            The relevant axis data for this model.
        initialConditions : ndarray[complex]
            The initial conditions of the single-time correlation functions.
        """

        # Ensure the initial conditions are a complex numpy array so that the ODE solver
        # solves in the complex domain.
        initialConditions = np.array(initialConditions, dtype=complex)

        # Stores the axes.
        self.__axes = axes

        # Stores parameters to be shared amongst the single- and double-time correlations by the ODE solver.
        odeParams = {
            'fun' : self.__ClassicallyDrivenSSHEquations,
            'rtol' : 1e-10,
            'atol' : 1e-12,
            'max_step' : 0.01 / self.__params.decayConstant
        }

        self.__correlationData.singleTime = self.__CalculateSingleTimeCorrelations(initialConditions, odeParams)

        self.__correlationData.singleFourierSeries = self.__CalculateSingleTimeFourierSeries(numPeriods = 10)

        self.__correlationData.doubleTime = self.__CalculateDoubleTimeCorrelations(odeParams)

    def __CalculateSteadyStateMask(self, numPeriods: int=None) -> np.ndarray[bool]:
        """
        Calculates a mask for the tau-axis to isolate the steady state part. Can also isolate a specific number
        of periods rather than the entire steady state.

        Parameters
        ----------
        numPeriods : int
            The number of periods in the steady state to isolate in the mask.
            If left as None, the mask will isolate the entire steady state.

        Returns
        -------
        ndarray[bool]
            A mask that will isolate the desired part of the steady state.
        """

        if numPeriods is None:
            return self.__axes.tauAxisDim >= self.__axes.steadyStateCutoff
        
        else:
            dimPeriod = self.__params.decayConstant / self.__params.drivingFreq
            return (self.__axes.steadyStateCutoff <= self.__axes.tauAxisDim) & (self.__axes.tauAxisDim <= self.__axes.steadyStateCutoff + dimPeriod * numPeriods)

    def __CalculateSingleTimeCorrelations(self, initialConditions: np.ndarray[complex], odeParams: dict) -> np.ndarray[complex]:
        r"""
        Calculates the single-time correlation functions.

        Parameters
        ----------
        initialConditions : ndarray[complex]
            The initial conditions of the single-time correlation functions.
        odeParams: dict
            A dictionary containing the relevant parameters for the function which solves the ODE.

        Returns
        -------
        ndarray[complex]
            An array of shape (3, tauAxis.size) which contains the value of the correlation functions at each time.
            The indices correspond to 0 = $\langle \sigma_-(t) \rangle$, 1 = $\langle \sigma_+(t) \rangle$, 2 = $\langle \sigma_z(t) \rangle$.
        """
 
        # Solves the single time solutions.
        inhomPart = -self.__params.decayConstant

        return integrate.solve_ivp(
            t_span = np.array([0, np.max(self.__axes.tauAxisSec)]),
            t_eval = self.__axes.tauAxisSec,
            y0 = initialConditions,
            args = (inhomPart,),
            **odeParams
        ).y
    
    def __CalculateSingleTimeFourierSeries(self, numPeriods: int=10) -> list[Fourier]:
        """
        Calculates the fourier series of the single-time correlation functions in the steady state.

        Parameters
        ----------
        numPeriods : int
            The number of steady-state periods to use when calculating the Fourier series.

        Returns
        -------
        list[Fourier]
            A list of the fourier series for each correlation function, where the indices 0, 1, and 2 correspond
            to the operators with subscripts -, +, and z respectively.
        """

        fourierMask = self.__CalculateSteadyStateMask(numPeriods)
        fourierSeries = []
        for i in range(3):
            fourierSeries.append(Fourier.FromSamples(
                baseFreq = self.__params.drivingFreq,
                y = self.__correlationData.singleTime[i, fourierMask],
                x = self.__axes.tauAxisSec[fourierMask],
                numPeriods = numPeriods
            ))

        return fourierSeries
    
    def __CalculateDoubleTimeCorrelations(self, odeParams: dict) -> np.ndarray[complex]:
        r"""
        Calculates the double-time correlation functions.

        Parameters
        ----------
        odeParams: dict
            A dictionary containing the relevant parameters for the function which solves the ODE.

        Returns
        -------
        ndarray[complex]
            The double-time correlations, of shape (3, 3, tAxis.size, tauAxis.size), where the first and second
            index correspond to the left and right operator respectively, with the same indexing as described in
            __CalculateSingleTimeFourierSeries().
        """

        doubleTime = np.zeros((3, 3, self.__axes.tAxisSec.size, self.__axes.tauAxisSec.size), dtype=complex)

        # Calculates the double-time initial conditions based on the single-time correlations for
        # each time within the steady-state period that we want to calculate.
        doubleTimeInitialConditions = self.__CalculateDoubleTimeInitialConditions()
        # Calculates the inhomogenous parts for each double-time initial condition.
        doubleTimeInhomParts = self.__CalculateDoubleTimeInhomogeousParts()

        # Loops through each initial condition time t.
        for tIndex, t in enumerate(self.__axes.tAxisSec):
            # Loops through all 3 operators that we can left-multiply by.
            for i in range(3):
                # Solves system.
                doubleTime[i, :, tIndex, :] = integrate.solve_ivp(
                    t_span = t + np.array([0, np.max(self.__axes.tauAxisSec)]),
                    t_eval = t + self.__axes.tauAxisSec,
                    y0 = doubleTimeInitialConditions[i, :, tIndex],
                    args = (doubleTimeInhomParts[i, tIndex],),
                    **odeParams
                ).y

        return doubleTime

    def __CalculateDoubleTimeInitialConditions(self) -> np.ndarray[complex]:
        """
        Calculates the double-time initial conditions for all 9 double-time correlators, at each time t.

        Returns
        -------
        ndarray[complex]
            An array of shape (3, 3, tAxis.size), where the first two dimensions correspond to the left and right operator
            in the double-time correlations, and the third corresponds to which point in the steady-state period along the
            t-axis we have chosen as our initial condition.
        """

        return np.array([
            # When left-multiplying by $\sigma_-(t)$
            [
                np.zeros(self.__axes.tAxisSec.size, dtype=complex),
                -0.5 * (self.__correlationData.singleFourierSeries[2].Evaluate(self.__axes.tAxisSec) - 1),
                self.__correlationData.singleFourierSeries[0].Evaluate(self.__axes.tAxisSec)
            ],
            # When left-multiplying by $\sigma_+(t)$
            [
                0.5 * (self.__correlationData.singleFourierSeries[2].Evaluate(self.__axes.tAxisSec) + 1),
                np.zeros(self.__axes.tAxisSec.size, dtype=complex),
                -self.__correlationData.singleFourierSeries[1].Evaluate(self.__axes.tAxisSec)
            ],
            # When left-multiplying by $\sigma_z(t)$
            [
                -self.__correlationData.singleFourierSeries[0].Evaluate(self.__axes.tAxisSec),
                self.__correlationData.singleFourierSeries[1].Evaluate(self.__axes.tAxisSec),
                np.ones(self.__axes.tAxisSec.size, dtype=complex),
            ]], dtype=complex
        )
    
    def __CalculateDoubleTimeInhomogeousParts(self) -> np.ndarray[complex]:
        """
        Calculates the inhomogenous parts for all 3 possible operators that we left-multiply by
        when calculating the double-time correlations.

        Returns
        -------
        ndarray[complex]
            An array of shape (3, tAxis.size) corresponding to the operator that we are left-multiplying
            by (axis 0) at the time on the t-axis that we have chosen as our initial condition (axis 1).
        """

        inhomParts = np.zeros((3, self.__axes.tAxisSec.size), dtype=complex)

        for i in range(3):
            inhomParts[i, :] = self.__correlationData.singleFourierSeries[i].Evaluate(self.__axes.tAxisSec) * -self.__params.decayConstant

        return inhomParts
    
    def CalculateCurrent(self) -> None:
        """Calculates all of the relevant attributes for the current."""

        # Calculates current operator in time- and frequency- domain.
        self.__currentData.timeDomainCurrent, self.__currentData.freqDomainCurrent = self.__CalculateCurrentExpectation()

        # Calculates fourier series for time-domain current operator.
        self.__currentData.currentFourierSeries = self.__CalculateCurrentFourier()

        # Calculates double-time current operator.
        self.__currentData.doubleTimeCurrent = self.__CalculateDoubleTimeCurrent()

        # Calculates the integrated, connected double-time current operator, plus its components (which are also integrated).
        self.__currentData.integratedDoubleTimeCurrent, self.__currentData.doubleTimeCurrentProduct, self.__currentData.timeConnectedCorrelator = self.__CalculateConnectedCurrentCorrelator()

        # Calculates the FFT of the integrated connected double-time current correlator in steady state.
        mask = self.__CalculateSteadyStateMask()
        self.__currentData.freqConnectedCorrelator = np.fft.fftshift(np.fft.fft(self.__currentData.timeConnectedCorrelator[mask]))

        # Calculates the Fourier transform of the connected correlator at the harmonics of the driving frequency.
        self.__currentData.harmonics = self.__CalculateCurrentHarmonics()

    def __CalculateCurrentExpectation(self) -> tuple[np.ndarray[complex], np.ndarray[complex]]:
        """Calculates the expectation of the current operator in the time- and frequency-domains.
        
        Returns
        -------
        ndarray[complex]
            The current operator evaluated at each time in the tau-axis.
        ndarray[complex]
            The FFT of the current operator in the steady state.
        """

        # Defines useful terms.
        drivingSamples = self.__SinusoidalDrivingTerm(self.__axes.tauAxisSec)

        coeff1 = -np.sin(self.__params.k - self.__params.phiK - drivingSamples)
        coeff2 = 1j * np.cos(self.__params.k - self.__params.phiK - drivingSamples)

        operator1 = self.__correlationData.singleTime[2]
        operator2 = self.__correlationData.singleTime[1] - self.__correlationData.singleTime[0]

        # Calculates the current operator in terms of the previously calculated expectation values.
        timeData = self.__params.t2 * (coeff1 * operator1 + coeff2 * operator2)

        mask = self.__CalculateSteadyStateMask()
        freqData = np.fft.fftshift(np.fft.fft(timeData[mask]))

        return timeData, freqData
    
    def __CalculateCurrentFourier(self) -> Fourier:
        """
        Calculates the fourier series of the current operator.

        Returns
        -------
        Fourier
            The Fourier series of the expectation of the current operator
            in its steady state.
        """

        r"""
        We have already expressed the current operator in terms of the Pauli matrices.
        
        $\langle j(t) \rangle = \sum_i j_i (t) \langle \sigma_i(t) \rangle$

        where $i \in \{ -, +, z \}$. We want to find the fourier series such that

        $j_i (t) = \sum_{n = -N}^N j_{i, n} e^{i\omega n t}$

        which we have solved analytically.
        """

        # Calculates the analytically derived Fourier series for each
        # of the coefficients of the current operator, when its expectation is written out
        # in terms of the pauli matrices.
        n = Fourier.DetermineMaxN(self.__axes.tauAxisSec, self.__params.drivingFreq)
        coefficients = np.zeros((3, 2 * n + 1), dtype=complex)

        # Defines a useful constant.
        theta = self.__params.k - self.__params.phiK

        # Calculates each of the coefficients for the fourier series.
        for i in np.arange(-n, n + 1):
            # Coefficient for $j_-(t)$.
            coefficients[0, i + n] = -0.5j * self.__params.t2 * special.jv(i, self.__params.drivingAmplitude) * (float(-1)**i * np.exp(1j * theta) + np.exp(-1j * theta))
            
            # Coefficient for $j_z(t)$.
            coefficients[2, i + n] = 0.5j * self.__params.t2 * special.jv(i, self.__params.drivingAmplitude) * (float(-1)**i * np.exp(1j * theta) - np.exp(-1j * theta))

        # Using $j_+(t) = -j_-(t)$, we can calculate the remaining coefficients.
        coefficients[1, :] = -coefficients[0, :]

        # Stores the fourier series of each coefficient in a list.
        coefficientFourierSeries = []
        for i in range(3):
            coefficientFourierSeries.append(
                Fourier(
                    self.__params.drivingFreq,
                    coefficients[i, :]
                )
            )

        r"""
        Now, we know the Fourier series for each operator $\langle \sigma_i(t) \rangle$ and each coefficient $j_i(t)$. Hence, for each term $j_i(t) \langle \sigma_i (t) \rangle$,
        we can express it as the convolution (or, product of two Fourier series)

        $\langle j(t) \rangle = \sum_i \left[ \left( \sum_{n = -N}^N j_{i, n} e^{in \omega t} \right) \left( \sum_{n = -N}^N \sigma_{i, n} e^{in \omega t} \right) \right]$

        Using the Fourier class, we can calculate the fourier transform of the current operator easily.
        """

        # Convolves each current coefficient's Fourier series with the corresponding operator's Fourier series.
        for i in range(3):
            coefficientFourierSeries[i] = Fourier.Convolve(coefficientFourierSeries[i], self.__correlationData.singleFourierSeries[i])

        # Now, we can simply sum up the three fourier series for each value of i, giving us the fourier series
        # for the expectation of the current.
        return np.sum(coefficientFourierSeries)
    
    def __CalculateDoubleTimeCurrent(self) -> np.ndarray[complex]:
        r"""
        Calculates the double-time current expectation.
        
        Returns
        -------
        ndarray[complex]
            An array of the shape (tAxis.size, tauAxis.size), containing the value of the double-time operator
            evaluated at times $t$ and $t' = t + \tau$.
        """

        # Defines useful properties.
        theta = self.__params.k - self.__params.phiK
        # Forms the driving samples for $A(t)$, with shape (tAxis.size)
        drivingSamplesT = self.__SinusoidalDrivingTerm(self.__axes.tAxisSec)
        
        # Forms a matrix of shape (tAxis.size, tauAxis.size) where each entry
        # corresponds to the value $A(t + \tau)$.
        drivingSamplesTau = np.add.outer(self.__axes.tAxisSec, self.__axes.tauAxisSec)
        drivingSamplesTau = self.__SinusoidalDrivingTerm(drivingSamplesTau)

        # Calculates the coefficients, which each have shape (tAxis.size, tauAxis.size).
        # Since drivingSamplesT has shape (tAxis.size), but drivingSamplesTau has size (tAxis.size, tauAxis.size),
        # we give drivingSamplesT a fake second axis, so that as we're iterating through the values of $\tau$ in $A(t + \tau)$,
        # we are moving through the fake second axis of $A(t)$, and so each of the $A(t)$ terms are matched up with the $A(t + \tau)$ terms.

        # Basically, we give drivingSamplesT a fake second axis to match their shapes.
        coeff1 = np.sin(theta - drivingSamplesT[:, np.newaxis]) * np.sin(theta - drivingSamplesTau)
        coeff2 = -np.cos(theta - drivingSamplesT[:, np.newaxis]) * np.cos(theta - drivingSamplesTau)
        coeff3 = -1j * np.cos(theta - drivingSamplesT[:, np.newaxis]) * np.sin(theta - drivingSamplesTau)
        coeff4 = -1j * np.sin(theta - drivingSamplesT[:, np.newaxis]) * np.cos(theta - drivingSamplesTau)

        # Passes the coefficients along to the diagnostic data for debugging use.
        self.__diagnosticData.CalculateConnectedCorrelatorTerms(self.__axes,
                                                                self.__params,
                                                                np.array([coeff1, coeff2, coeff3, coeff4]),
                                                                self.__correlationData)

        # Calculates all of the operator terms that correspond to each coefficient, again with
        # shape (tAxis.size, tauAxis.size)
        operators1 = self.__correlationData.doubleTime[2, 2]
        operators2 = self.__correlationData.doubleTime[1, 1] - self.__correlationData.doubleTime[1, 0] - self.__correlationData.doubleTime[0, 1] + self.__correlationData.doubleTime[0, 0]
        operators3 = self.__correlationData.doubleTime[1, 2] - self.__correlationData.doubleTime[0, 2]
        operators4 = self.__correlationData.doubleTime[2, 1] - self.__correlationData.doubleTime[2, 0]

        return self.__params.t2**2 * (coeff1 * operators1 + coeff2 * operators2 + coeff3 * operators3 + coeff4 * operators4)
    
    def __CalculateConnectedCurrentCorrelator(self) -> tuple[np.ndarray[complex], np.ndarray[complex], np.ndarray[complex]]:
        r"""
        Calculates the connected current correlator, integrated over a steady state period
        w.r.t. $t$.
        
        Returns
        -------
        ndarray[complex]
            An array with shape (tauAxis.size,) containing the value of the integrated double-time
            current correlator

            $\frac{1}{T} \int dt\, \langle j(t) j(t + \tau) \rangle$

        ndarray[complex]
            An array with shape (tauAxis.size,) containing
            the value of the product of current expectations,
            
            $\frac{1}{T} \int dt\, \langle j(t) \rangle \langle j(t + \tau) \rangle$.

        ndarray[complex]
            The value of the integrated double-time connected current correlation.

            $\frac{1}{T} \int dt\, \langle j(t) j(t + \tau) \rangle - \langle j(t) \rangle \langle j(t + \tau) \rangle$
        """

        # Calculates the integrated product of expectations using the analytically derived
        # Fourier series identity.
        coeffs = np.abs(self.__currentData.currentFourierSeries.coeffs)**2
        doubleTimeCurrentProduct = Fourier(
            baseFreq = self.__params.drivingFreq,
            coeffs = coeffs
        ).Evaluate(self.__axes.tauAxisSec)

        # For debugging purposes, calculates the above term manually by numerically multiplying the
        # current expectation fourier series together at each time $t$ and $t + \tau$.
        
        # Calculates $\langle j(t) \rangle$, with shape (tAxis.size)
        currentT = self.__currentData.currentFourierSeries.Evaluate(self.__axes.tAxisSec)
        # Calculates $\langle j(t + \tau) \rangle$, with shape (tAxis.size, tauAxis.size)
        currentTau = np.add.outer(self.__axes.tAxisSec, self.__axes.tauAxisSec)
        currentTau = self.__currentData.currentFourierSeries.Evaluate(currentTau)

        # Multiplies the two together, calculating $\langle j(t) \rangle \langle j(t + \tau) \rangle$
        # with size (tAxis.size, tauAxis.size).
        numericalTerm = currentT[:, np.newaxis] * currentTau
        
        # Numerically integrates the data. Sets the value in current data within the function since its
        # just used for debugging, so its not worth returning.
        self.__diagnosticData.numericalDoubleTimeCurrentProduct = self.__params.drivingFreq * np.trapezoid(
            y = numericalTerm,
            x = self.__axes.tAxisSec,
            axis = 0
        )

        # Then, integrates the double-time current to average out over a steady-state period
        # and calculates the connected correlator.
        integratedDoubleTimeData = self.__params.drivingFreq * np.trapezoid(
            y = self.__currentData.doubleTimeCurrent,
            x = self.__axes.tAxisSec,
            axis = 0
        )

        connectedCorrelator = integratedDoubleTimeData - doubleTimeCurrentProduct

        return integratedDoubleTimeData, doubleTimeCurrentProduct, connectedCorrelator
    
    def __CalculateCurrentHarmonics(self, maxHarmonic: int=12) -> np.ndarray[complex]:
        r"""
        Calculates the Fourier transform of the integrated, connected double-current conected correlator
        at the integer multiples of the driving frequency.

        Parameters
        ----------
        maxHarmonic : int
            The number of harmonics to calculate. The function will calculate the harmonics from
            -maxHarmonics to maxHarmonics.

        Returns
        -------
        An array of shape (2n + 1,), which contains the Fourier transforms at the frequencies
        $2\pi k f$ for $k \in [-n, n] \subset \mathbb{Z}$.
        """

        angularFreq = 2 * np.pi * self.__params.drivingFreq
        harmonics = np.zeros((2 * maxHarmonic + 1), dtype=complex)

        # Gets the data in steady-state, we will not consider data not in steady state.
        mask = self.__CalculateSteadyStateMask()
        steadyStateTauAxis = self.__axes.tauAxisSec[mask]
        steadyStateConnectedCorrelator = self.__currentData.timeConnectedCorrelator[mask]

        # Creates the required exponential terms. First axis is degree of harmonic, second axis is tau axis.
        expTerms = np.outer(-1j * np.arange(-maxHarmonic, maxHarmonic + 1) * angularFreq, steadyStateTauAxis)
        expTerms = np.exp(expTerms)

        # We want to multiply each fixed n with the value of the connected correlator at every value of tauaxis.
        # Hence, we multiply each row (fixed harmonic) by the connected correlator.
        integrand = expTerms * np.stack((steadyStateConnectedCorrelator,) * expTerms.shape[0], axis=0)

        # Now, our integrand contains every relevant term $D_k(\tau) e^{-in \omega \tau}$, with the first axis determining n, and the second
        # determining $\tau$. Hence, since we are integrating along the tau axis to determine the magnitude at each harmonic, we integrate
        # along the tau axis (axis 1).
        harmonics = integrate.simpson(
            y = integrand,
            x = steadyStateTauAxis,
            axis = 1
        )

        return harmonics
    
    @property
    def currentData(self) -> CurrentData:
        return self.__currentData
    
    @property
    def correlationData(self) -> CorrelationData:
        return self.__correlationData
    
    @property
    def diagnosticData(self) -> DiagnosticData:
        return self.__diagnosticData