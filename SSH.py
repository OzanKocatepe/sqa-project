import numpy as np
import scipy.integrate as integrate
import scipy.special as special
from typing import Callable, Any
import math

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
        self._tAxis = None
        self._drivingTerm = None
        self._solution = None
        self._currentTime = None
        self._currentFreq = None
        self._freqAxis = None 
 
    @property
    def solution(self) -> Any:
        """Gets the solution for the expectations.
        
        Returns
        -------
        Any
            The object returned by scipy.integrate.solve_ivp which contains the solutions
            to the ODE.
            
        Raises
        ------
        ValueError
            If Solve() has not been called, so no solution has been calculated yet.
        """
        if self._solution is None:
            raise ValueError("Call Solve() first.")
        else:
            return self._solution
        
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

    def ClassicallyDrivenSSHEquations(self, t: float, c: np.ndarray[float], A: Callable[[np.typing.ArrayLike], np.typing.ArrayLike]) -> np.ndarray[float]:
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
        d = np.array([0, 0, -self.decayConstant], dtype=complex)
    
        return B @ c + d

    def Solve(self, tAxis: np.ndarray[float], initialConditions: np.ndarray[complex], drivingTerm: Callable[[float], float]=None) -> Any:
        r"""
        Solves the system of ODEs for the expectations of our two-level system operators $(\langle \sigma_-(t) \rangle,\, \langle \sigma_+(t) \rangle,\, \langle \sigma_z(t) \rangle)$.
            
        Parameters
        ----------
        tAxis : ndarray[float]
            The points in time (in units of $\gamma_-^{-1}$) that the final solution should be evaluated at.
        initialConditions : ndarray[complex]
            The initial conditions of the expectations.
        drivingTerm : Callable[[float], float]
            The classical driving term that will be used in the modelling of the system.
            By default, it is a sinusoidal driving term.

        Returns
        -------
        Any
            The object returned by scipy.integrate.solve_ivp which contains the solutions
            to the ODE.
        """

        if drivingTerm is None:
            self._drivingTerm = self.SinusoidalDrivingTerm
        else:
            self._drivingTerm = drivingTerm

        # Internally stores the tAxis in seconds. Whenever the tAxis is inputted or outputted,
        # it will be given in units of $\gamma_-^{-1}$, but internally we always use absolute seconds.
        self._tAxis = tAxis / self.decayConstant
  
        self._solution = integrate.solve_ivp(fun = self.ClassicallyDrivenSSHEquations,
                                             t_span = np.array([np.min(self._tAxis), np.max(self._tAxis)]),
                                             y0 = initialConditions,
                                             t_eval = self._tAxis,
                                             rtol = 1e-10,
                                             atol = 1e-12,
                                             args = (self._drivingTerm,))
        
        return self._solution
        
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

        if self._solution is None:
            raise ValueError("Call Solve() first.")

        # Defines useful terms.
        Ek = self.t1 + self.t2 * np.exp(1j * self.k)
        phiK = np.angle(Ek)
        drivingSamples = self._drivingTerm(self._tAxis)

        # Calculates the current operator in terms of the previously calculated expectation values.
        self._currentTime = self.t2 * (
            -np.sin(self.k - phiK - drivingSamples) * self._solution.y[2]
            + 1j * np.cos(self.k - phiK - drivingSamples) * (self._solution.y[1] - self._solution.y[0])
        )

        # Only considers the system in steady state for the Fourier transform, if desired.
        if steadyStateCutoff != None:
            mask = self._tAxis >= steadyStateCutoff / self.decayConstant
        else:
            mask = np.full(self._tAxis.size, True, dtype=bool)

        steadyStateAxis = self._tAxis[mask]

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

        if self._solution is None:
            raise ValueError("Run Solve() first.")

        # Creates an array of shape (3, 2n + 1) such that the first dimension corresponds to which correlation function
        # we want, and the second corresponds to which coefficient we want, from -n to n.
        # Since the indices go from 0 to 2n, the index for coefficient $c_i$ is actually $n + i$.
        coefficients = np.zeros((3, 2 * n + 1), dtype=complex)

        # Creates a mask over one period in steady state.
        periodMask = (steadyStateCutoff / self.decayConstant <= self._tAxis) & (self._tAxis <= steadyStateCutoff / self.decayConstant + numPeriods / self.drivingFreq)

        # Loops through which function we are looking at.
        for functionIndex in np.arange(3):
            print(f"Calculating coefficients of function {functionIndex}...")

            # Defines the t and f(t) arrays only within the mask.
            tWindow = self._tAxis[periodMask]
            fWindow = self._solution.y[functionIndex][periodMask]

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

        dx = np.mean(np.diff(self._tAxis))
        maxN = math.floor(1 / (4 * dx * np.pi * self.drivingFreq))

        if n is None:
            n = maxN
        else:
            n = math.min(n, maxN)

        coefficients = np.zeros((4 * n + 1), dtype=complex)
        expectationCoeff = self._CalculateExpectationCoefficients(n, steadyStateCutoff, numPeriods)
        currentCoeff = self._CalculateCurrentCoefficients(n)

        for m in range(-2 * n, 2 * n + 1):
            if m >= 0:
                lowerBound = m - n
                upperBound = n
            else:
                lowerBound = -n
                upperBound = m + n

            for k in range(lowerBound, upperBound + 1):
                coefficients[m + 2 * n] += expectationCoeff[0, k + n] * currentCoeff[0, (m + n) - (k + n)] + expectationCoeff[1, k + n] * currentCoeff[1, (m + n) - (k + n)] + expectationCoeff[2, k + n] * currentCoeff[2, (m + n) - (k + n)]

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
            An array of shape (tAxis.size,) with the value of the fourier expansion at each time t.

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
        expTerms = np.zeros((2 * n + 1, self._tAxis.size), dtype=complex)
        for t in range(self._tAxis.size):
            expTerms[:, t] = np.exp(1j * freq * np.arange(-n, n + 1) * self._tAxis[t])

        # Evaluates the value of the fourier expansion at each time t.
        fourierExpansion = np.zeros((self._tAxis.size,), dtype=complex)
        for t in range(self._tAxis.size):
            fourierExpansion[t] = np.dot(coefficients, expTerms[:, t])

        return fourierExpansion