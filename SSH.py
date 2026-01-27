import numpy as np
import scipy.integrate as integrate
import scipy.special as special
from typing import Callable
import math
from tqdm import tqdm

class SSH:
    """
    Contains the core physics and calculations for the 1-dimensional SSH model.
    Since we consider a fixed momentum, this system simplifies to essentially a two-level system.
    """

    def __init__(self, k: float, t1: float, t2: float, decayConstant: float, drivingAmplitude: float, drivingFreq: float):
        """
        Constructs an instance of the SSH model.

        Parameters
        ----------
        k : float
            The momentum.
        t1 : float
            The intracell(?) hopping amplitude.
        t2 : float
            The intercell(?) hopping amplitude.
        decayConstant: float
            The decay constant, in units of Hz.
        drivingAmplitude : float
            The amplitude of the classical driving term.
        drivingFreq : float
            The frequency of the driving term, in Hz.
        """

        self.k = k
        self.t1 = t1
        self.t2 = t2
        self.decayConstant = decayConstant
        self.drivingAmplitude = drivingAmplitude
        self.drivingFreq = drivingFreq

        # Defines the internal values that haven't been calculated yet.
        self._tauAxis = None
        self._steadyStatePeriodAxis = None
        self._drivingTerm = None
        self._singleTimeSolution = None
        self._doubleTimeSolution = None
        self._currentTime = None
        self._currentFreq = None
        self._freqAxis = None 

    @property
    def steadyStatePeriodAxis(self) -> np.ndarray[float]:
        r"""
        Gets the values of time that we take to be initial conditions
        of the double-time correlation functions.

        Returns
        -------
        ndarray[float]
            The values of t, in units of $\gamma_-^{-1}$ that we use as the initial conditions when calculating
            the double-time correlations. These should represent a single period
            in the steady-state.

        Raises
        ------
        ValueError
            If Solve() has not been called, so no steadyStateCutoff has been given yet.
        """

        if self._steadyStatePeriodAxis is None:
            raise ValueError("Call Solve() first.")
        else:
            return self._steadyStatePeriodAxis
 
    @property
    def singleTimeSolution(self) -> np.ndarray[complex]:
        """Gets the solution for the single-time correlation functions.
        
        Returns
        -------
        ndarray[complex]
            An array of shape (3, tAxis.size) which contains the single-time correlation functions.
            
        Raises
        ------
        ValueError
            If Solve() has not been called, so no solution has been calculated yet.
        """

        if self._singleTimeSolution is None:
            raise ValueError("Call Solve() first.")
        else:
            return self._singleTimeSolution

    @property
    def doubleTimeSolution(self) -> np.ndarray[complex]:
        """Gets the solution for the double-time correlation functions.
        
        Returns
        -------
        ndarray[complex]
            An array of shape (3, 3, steadyStateIndices.size, tAxis.size) which contains the
            double-time correlation functions.
            
        Raises
        ------
        ValueError
            If Solve() has not been called, so no solution has been calculated yet.
        """
        if self._doubleTimeSolution is None:
            raise ValueError("Call Solve() first.")
        else:
            return self._doubleTimeSolution
        
    @property
    def currentTime(self) -> np.ndarray[complex]:
        """Gets the current operator in the time domain.
        
        Returns
        -------
        ndarray[complex]
            The values of the current operator at each time in the given tAxis.
            
        Raises
        ------
        ValueError
            If CalculateCurrent() has not been called, meaning the current operator hasn't been calculated yet.
        """

        if self._currentTime is None:
            raise ValueError("Call CalculateCurrent() first.")
        else:
            return self._currentTime

    @property
    def currentFreq(self) -> np.ndarray[float]:
        """Gets the current operator in the frequency domain.
        
        Returns
        -------
        ndarray[float]
            The amplitudes of the fourier transform of the current operator at each frequency in the freqAxis.
            
        Raises
        ------
        ValueError
            If CalculateCurrent() has not been called, meaning the current operator hasn't been calculated yet.
        """

        if self._currentFreq is None:
            raise ValueError("Call CalculateCurrent() first.")
        else:
            return self._currentFreq

    @property
    def freqAxis(self) -> np.ndarray[float]:
        """Gets the frequency axis.
        
        Returns
        -------
        ndarray[float]
            The frequencies that correspond to the amplitudes of the Fourier transform of the current operator
            found in CalculateCurrent().
            
        Raises
        ------
        ValueError
            If CalculateCurrent() has not been called, meaning the current operator hasn't been calculated yet.
        """

        if self._freqAxis is None:
            raise ValueError("Call CalculateCurrent() first.")
        else:
            return self._freqAxis

    def SinusoidalDrivingTerm(self, t: np.typing.ArrayLike) ->  np.typing.ArrayLike:
        """A classical, sinusoidal driving term for the system.
        
        Parameters
        ----------
        t : ArrayLike
            The points in time (in seconds) to evaluate the driving term at.
            
        Returns
        -------
        ArrayLike
            The value of the driving term at each time t.
        """

        return self.drivingAmplitude * np.sin(2 * np.pi * self.drivingFreq * t)

    def ClassicallyDrivenSSHEquations(self, t: float, c: np.ndarray[float], A: Callable[[np.typing.ArrayLike], np.typing.ArrayLike], inhomPart: float) -> np.ndarray[float]:
        r"""
        The ODE (equations of motion) for the single-time expectations of $\sigma_-(t)$, $\sigma_+(t)$, and $\sigma_z(t)$. 

        Parameters
        ----------
        t : float
            The time t, in seconds.
        c : ndarray[float]
            The 3-component single-time correlation array.
        A : Callable[[np.typing.ArrayLike], np.typing.ArrayLike]
            The classical driving term. Should be a vectorisable function of t.
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
        Ek = self.t1 + self.t2 * np.exp(1j * self.k)
        phiK = np.angle(Ek)
        vZ = 2 * self.t2 * np.sin(self.k - phiK - 0.5 * A(t)) * np.sin(0.5 * A(t))
        vPm = 2j * self.t2 * np.cos(self.k - phiK - 0.5 * A(t)) * np.sin(0.5 * A(t))

        # Defines the coefficient matrix.
        B = np.array([[-2j * (np.abs(Ek) + vZ) - 0.5 * self.decayConstant  , 0                                                  ,  -1j * vPm            ],
                      [0                                                   , 2j * (np.abs(Ek) + vZ) - 0.5 * self.decayConstant  ,  -1j * vPm            ],
                      [2j * vPm                                            , 2j * vPm                                           ,  -self.decayConstant  ]], dtype=complex)
    
        # Inhomogenous part.
        d = np.array([0, 0, inhomPart], dtype=complex)
    
        return B @ c + d

    def Solve(self, tauAxis: np.ndarray[float], initialConditions: np.ndarray[complex], drivingTerm: Callable[[float], float]=None, steadyStateCutoff: float=25, debug: bool=False) -> tuple[np.ndarray[complex], np.ndarray[complex]]:
        r"""
        Solves the system of ODEs for the expectations of our two-level system operators $(\langle \sigma_-(t) \rangle,\, \langle \sigma_+(t) \rangle,\, \langle \sigma_z(t) \rangle)$, and the double-time correlation functions $\langle \sigma_i (t) \sigma_j (t + \tau) \rangle$.
            
        Parameters
        ----------
        tauAxis : ndarray[float]
            The offsets from the initial time $t$ that the functions should be calculated on. The double-time correlations must
            be defined up to the steady state cutoff + one period + the entire tau axis, so we need to calculate them for
            more points than just the tau axis.
        initialConditions : ndarray[complex]
            The initial conditions of the single-time correlation functions. This is used to calculate
            the initial conditions of the double-time correlations.
        drivingTerm : Callable[[float], float]
            The classical driving term that will be used in the modelling of the system.
            By default, it is a sinusoidal driving term.
        steadyStateCutoff : float
            The time, in units of $\gamma_-^{-1}$, after which we consider the system in steady-state. This is used
            to calculate the initial conditions for the double-time correlations.
        debug : bool
            Whether to print out debug progress statements.

        Returns
        -------
        ndarray[complex]
            An array of shape (3, tAxis.size) which contains the single-time correlation functions. The index of the first dimension
            corresponds to the correlation function, with 0 = $\langle \sigma_-(t) \rangle$, 1 = $\langle \sigma_+(t) \rangle$, and 2 = $ \langle \sigma_z(t) \rangle$.
        ndarray[complex]
            An array of shape (3, 3, steadyStateIndices.size, tAxis.size) which contains the double-time correlation functions.
            The first and second dimensions correspond to the left and right operators in the correlation function respectively,
            following the index convention for the single-time correlatinos. The left operator is evaluated at time $t$, whereas the right operator is evaluated at time $t' = t + \tau$.
            The third dimension corresponds to different values of time within a steady-state period that we take to be the initial
            conditions, and the fourth dimension corresponds to the values of the function at different points of $\tau$ on the tAxis.
        """

        if drivingTerm is None:
            self._drivingTerm = self.SinusoidalDrivingTerm
        else:
            self._drivingTerm = drivingTerm

        # Internally stores the tauAxis in seconds. Whenever the tauAxis is inputted or outputted,
        # it will be given in units of $\gamma_-^{-1}$, but internally we always use absolute seconds.
        self._tauAxis = tauAxis / self.decayConstant

        # Calculates how much extra domain is needed for the single time correlations such that they
        # extend to the steady state cutoff + 1 period + the largest value of tau (+ some buffer).
        # Generates the necessary extra points with the mean spacing of the given tau axis.
        spacing = np.mean(np.diff(self._tauAxis))
        extraDomain = np.arange(tauAxis[-1], steadyStateCutoff / self.decayConstant + 10 / self.drivingFreq + tauAxis[-1], spacing) + spacing
        singleTimeDomain = np.concatenate((tauAxis, extraDomain))
 
        # Stores the function parameters, since they are mostly the same for the single- and double-time
        # correlations.
        initialConditions = np.array(initialConditions, dtype=complex)
        params = {
            'fun' : self.ClassicallyDrivenSSHEquations,
            't_span' : np.array([np.min(singleTimeDomain), np.max(singleTimeDomain)]),
            'y0' : initialConditions,
            't_eval' : singleTimeDomain,
            'rtol' : 1e-10,
            'atol' : 1e-12,
            'args' : (self._drivingTerm, -self.decayConstant,)
        }
        self._singleTimeSolution = integrate.solve_ivp(**params).y

        # Calculates the indices of the first period of the steady-state, based on the given steadyStateCutoff.
        steadyStateIndices = np.where(
            (steadyStateCutoff / self.decayConstant <= self._tauAxis)
            & (self._tauAxis <= steadyStateCutoff / self.decayConstant + 1 / self.drivingFreq)
        )[0]
        
        # Stores the tau axis values that we are considering our initial conditions for t,
        # in units of $\gamma_-^{-1}$.
        self._steadyStatePeriodAxis = self._tauAxis[steadyStateIndices] * self.decayConstant

        # Defines the double time solutions. The first dimension corresponds to the left-hand operator,
        # the second corresponds to the right hand operator, the third dimension corresponds to
        # the different times within a steady-state period that we consider our initial conditions at, and
        # the fourth dimension corresponds to the value of our time offset $\tau$.
        self._doubleTimeSolution = np.zeros((3, 3, steadyStateIndices.size, self._tauAxis.size), dtype=complex)

        # Loops through each time in the steady-state period that we want to consider.
        iterable = steadyStateIndices
        if debug:
            print(f"Calculating double-time correlations for k = {self.k / np.pi:.2f}pi...")
            iterable = tqdm(steadyStateIndices)

        for tIndex in iterable:
            # Calculates the double-time initial conditions based on the single-time correlations.
            doubleTimeInitialConditions = np.array([
                # When left-multiplying by $\sigma_-(t)$
                [
                    0,
                    -0.5 * (self._singleTimeSolution[2, tIndex] - 1),
                    self._singleTimeSolution[0, tIndex]
                ],
                # When left-multiplying by $\sigma_+(t)$
                [
                    0.5 * (self._singleTimeSolution[2, tIndex] + 1),
                    0,
                    -self._singleTimeSolution[1, tIndex]
                ],
                # When left-multiplying by $\sigma_z(t)$
                [
                    -self._singleTimeSolution[0, tIndex],
                    self._singleTimeSolution[1, tIndex],
                    1
                ]], dtype=complex
            )

            # Loops through all 3 operators that we can left-multiply by.
            # Each loops calculates the 3 double-time correlations corresponding to that
            # left operator.
            for i in range(3):
                params['t_span'] = np.array([np.min(self._tauAxis), np.max(self._tauAxis)])
                params['t_eval'] = self._tauAxis
                params['y0'] = doubleTimeInitialConditions[i, :]
                # Calculates the new inhomogenous term.
                params['args'] = (self._drivingTerm, -self.decayConstant * self._singleTimeSolution[i, tIndex],)

                # Solves system.
                self._doubleTimeSolution[i, :, steadyStateIndices - steadyStateIndices[0], :] = integrate.solve_ivp(**params).y
        
        if debug:
            print('\n')
        
        return self._singleTimeSolution, self._doubleTimeSolution
        
    def CalculateCurrent(self, steadyStateCutoff: float=None) -> tuple[np.ndarray[complex], np.ndarray[complex]]:
        r"""Calculates the current operator for the given parameters.
        
        Parameters
        ----------
        steadyStateCutoff : float
            The time (in units of $\gamma_-^{-1}$) which we consider the system to be in steady-state.
            i.e. we only consider the Fourier transform of the system after this point.

        Returns
        -------
        np.ndarray[complex]
            The value of the current operator at each point along the time axis.
            Should be entirely real (within numerical uncertainty).
        np.ndarray[complex]
            The fourier transform of the current operator.

        Raises
        ------
        ValueError
            If Solve() hasn't been called yet, so the expectation values of our operators haven't been calculated.
            This makes it impossible to calculate the current operator.
        """

        if self._singleTimeSolution is None:
            raise ValueError("Call Solve() first.")

        # Defines useful terms.
        Ek = self.t1 + self.t2 * np.exp(1j * self.k)
        phiK = np.angle(Ek)
        drivingSamples = self._drivingTerm(self._tauAxis)

        # Defines a mask to only include single time solution values within the tau axis,
        # and ignore the extra values defined only for convenience with the double-time correlations.
        tauMask = np.arange(self._tauAxis.size)

        # Calculates the current operator in terms of the previously calculated expectation values.
        self._currentTime = self.t2 * (
            -np.sin(self.k - phiK - drivingSamples) * self._singleTimeSolution[2][tauMask]
            + 1j * np.cos(self.k - phiK - drivingSamples) * (self._singleTimeSolution[1][tauMask] - self._singleTimeSolution[0][tauMask])
        )

        # Only considers the system in steady state for the Fourier transform, if desired.
        if steadyStateCutoff != None:
            mask = self._tauAxis >= steadyStateCutoff / self.decayConstant
        else:
            mask = np.full(self._tauAxis.size, True, dtype=bool)

        steadyStateAxis = self._tauAxis[mask]

        # Calculates the Fourier transform of the solution.
        self._currentFreq = np.fft.fftshift(np.fft.fft(self._currentTime[mask]))

        # Calculates the relevant frequency axis.
        sampleSpacing = (np.max(steadyStateAxis) - np.min(steadyStateAxis)) / steadyStateAxis.size
        self._freqAxis = np.fft.fftshift(np.fft.fftfreq(steadyStateAxis.size, sampleSpacing))

        return self._currentTime, self._currentFreq
    
    def _CalculateExpectationCoefficients(self, n: int, steadyStateCutoff: float=15, numPeriods: int=10) -> np.ndarray[complex]:
        r"""Calculates the first n coefficients in the fourier expansion of the correlation functions.
        
        Parameters
        ----------
        n : int
            The number of coefficients to calculate. Must be a positive number, as the function calculates
            the coefficients in the range -n to n.
        steadyStateCutoff : float
            The time, in units of $\gamma_-^{-1}$, after which we assume the system is in its periodic steady-state.
        numPeriods : int
            The number of steady-state periods to use to calculate the fourier coefficients.

        Returns
        -------
        ndarray[complex]
            The coefficients of the fourier expansion. This has shape (3, 2n + 1), where the first dimension
            corresponds to the correlation function, and the second dimension corresponds to the coefficient,
            ranging from -n to n.

        Raises
        ------
        ValueError
            If the correlations have not been calculated.
        """

        if self._singleTimeSolution is None:
            raise ValueError("Run Solve() first.")

        # Creates an array of shape (3, 2n + 1) such that the first dimension corresponds to which correlation function
        # we want, and the second corresponds to which coefficient we want, from -n to n.
        # Since the indices go from 0 to 2n, the index for coefficient $c_i$ is actually $n + i$.
        coefficients = np.zeros((3, 2 * n + 1), dtype=complex)

        # Creates a mask over one period in steady state.
        periodMask = (steadyStateCutoff / self.decayConstant <= self._tauAxis) & (self._tauAxis <= steadyStateCutoff / self.decayConstant + numPeriods / self.drivingFreq)

        # Loops through which function we are looking at.
        for functionIndex in np.arange(3):
            # Defines the t and f(t) arrays only within the mask.
            tWindow = self._tauAxis[periodMask]
            fWindow = self._singleTimeSolution[functionIndex][periodMask]

            # Subtracts any mean from the function window.
            fWindowMean = np.mean(fWindow)
            fWindow -= fWindowMean

            # Defines useful values.
            scalingFactor = self.drivingFreq / numPeriods
            angularFreq = 2 * np.pi * self.drivingFreq

            # Loops through the coefficients.
            for i in np.arange(-n, n + 1):
                # Calculates the coefficient for i, and stores it in index i + n.
                coefficients[functionIndex, i + n] = scalingFactor * np.trapezoid(
                    y = fWindow * np.exp(-1j * angularFreq * i * tWindow),
                    x = tWindow
                )

            # Manually adds back the mean to the constant term in the expansion.
            coefficients[functionIndex, n] += fWindowMean

            print('\n')
        
        return coefficients
    
    def _CalculateCurrentCoefficients(self, n: int) -> np.ndarray[complex]:
        r"""
        Calculates the fourier coefficients corresponding to the coefficients of the operators in the current operator.
        i.e. the coefficients for $j_-(t), j_+(t), j_z(t)$.
        
        Parameters
        ----------
        n : int
            The number of coefficients to calculate. Must be a positive number, as the function calculates
            the coefficients in the range -n to n.

        Returns
        -------
        ndarray[complex]
            The coefficients of the fourier expansion. This has shape (3, 2n + 1), where the first dimension
            corresponds to the coefficient function ($j_-(t), j_+(t), j_z(t)$), and the second dimension corresponds to the subscript of the coefficient,
            ranging from -n to n.
        """

        coefficients = np.zeros((3, 2 * n + 1), dtype=complex)

        Ek = self.t1 + self.t2 * np.exp(1j * self.k)
        phiK = np.angle(Ek)
        theta = self.k - phiK
        angularFreq = 2 * np.pi * self.drivingFreq

        for i in np.arange(-n, n + 1):
            # Coefficient for $j_-(t)$.
            coefficients[0, i + n] = -0.5j * self.t2 * special.jv(i, self.drivingAmplitude) * (float(-1)**i * np.exp(1j * theta) + np.exp(-1j * theta))
            
            # Coefficient for $j_z(t)$.
            coefficients[2, i + n] = 0.5j * self.t2 * special.jv(i, self.drivingAmplitude) * (float(-1)**i * np.exp(1j * theta) - np.exp(-1j * theta))

        # Using $j_+(t) = -j_-(t)$, we can calculate the remaining coefficients.
        coefficients[1, :] = -coefficients[0, :]

        return coefficients
    
    def CalculateCurrentExpectationCoefficients(self, n: int=None, steadyStateCutoff=15, numPeriods=10) -> np.ndarray[complex]:
        r"""
        Calculates the coefficients for the fourier expansion of the expectation of the current operator
        into the laser harmonics. These are calculated from the fourier coefficients of the single-time
        correlations $\langle \sigma_-(t) \rangle,\, \langle \sigma_+(t) \rangle,\, \langle \sigma_z(t) \rangle$, and the current coefficient functions $j_-(t), j_+(t), j_z(t)$.

        Parameters
        ----------
        n : int
            The number of coefficients to calculate for the expectations and currents. Must be a positive number, as the function calculates
            the coefficients in the range -n to n. If unspecified, will use the maximum possible n such that
            the angular frequency is still under the Nyquist frequency.
        steadyStateCutoff : float
            The time, in units of $\gamma_-^{-1}$, after which we assume the system is in its periodic steady-state.
            This is the state which we will use to find the coefficients for the single-time correlations.
        numPeriods : int
            The number of steady-state periods to use to calculate the fourier coefficients of the single-time
            correlations.

        Returns
        -------
        ndarray[complex]
            The coefficients in the fourier expansion of the expectation of the current operator, from -2n to 2n.
        """

        dx = np.mean(np.diff(self._tauAxis))
        maxN = math.floor(1 / (4 * dx * np.pi * self.drivingFreq))

        if n is None:
            n = maxN
        else:
            n = math.min(n, maxN)

        coefficients = np.zeros((4 * n + 1), dtype=complex)
        expectationCoeff = self._CalculateExpectationCoefficients(n, steadyStateCutoff, numPeriods)
        currentCoeff = self._CalculateCurrentCoefficients(n)

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

        return coefficients

    
    def EvaluateFourierExpansion(self, coefficients: np.ndarray[complex], freq: float=None) -> np.ndarray[complex]:
        """
        Evaluates the fourier expansion of a function based on its coefficients.

        Parameters
        ----------
        coefficients : ndarray[complex]
            An array of shape (2 * n + 1,) for some integer n which contains all of the coefficients from -n to n of our function.
        freq : float
            The frequency that we will be expanding into harmonics of. By default, will using the
            *angular* driving frequency of this system.

        Returns
        -------
        ndarray[complex]
            An array of shape (tauAxis.size,) with the value of the fourier expansion at each time offset tau.

        Raises
        ------
        ValueError
            If the coefficients array has the wrong shape.
        """

        if freq is None:
            freq = 2 * np.pi * self.drivingFreq

        n = (coefficients.shape[0] - 1) // 2
        if 2 * n + 1 != coefficients.shape[0]:
            raise ValueError("coefficients must be an array of shape (2n + 1,) for some integral n.")

        # Generates the set of exponential terms at each time t.
        expTerms = np.zeros((2 * n + 1, self._tauAxis.size), dtype=complex)
        for t in range(self._tauAxis.size):
            expTerms[:, t] = np.exp(1j * freq * np.arange(-n, n + 1) * self._tauAxis[t])

        # Evaluates the value of the fourier expansion at each time tau.
        fourierExpansion = np.zeros((self._tauAxis.size,), dtype=complex)
        for t in range(self._tauAxis.size):
            fourierExpansion[t] = np.dot(coefficients, expTerms[:, t])

        return fourierExpansion