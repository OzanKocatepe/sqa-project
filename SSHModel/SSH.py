import numpy as np
import scipy.integrate as integrate
from typing import Callable
from tqdm import tqdm

from SSHModel import CorrelationData, CurrentData, SSHParameters, Fourier

class SSH:
    """
    Contains the core physics and calculations for the 1-dimensional SSH model.
    Since we consider a fixed momentum, this system simplifies to essentially a two-level system.
    """

    def __init__(self, k: float, params: SSHParameters):
        """
        Constructs an instance of the SSH model.

        Parameters
        ----------
        k : float
            The momentum.
        params: SSHParameters
            The parameters of the SSH model.
        """

        self.__params = params
        self.__correlationData: CorrelationData = None
        self.__currentData: CurrentData = None

    def __SinusoidalDrivingTerm(self, t: float | np.ndarray[float]) ->  float | np.ndarray[float]:
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

        return self.drivingAmplitude * np.sin(2 * np.pi * self.drivingFreq * t)

    def __ClassicallyDrivenSSHEquations(self, t: float, c: np.ndarray[float], inhomPart: float, A: Callable[[np.typing.ArrayLike], np.typing.ArrayLike]=None) -> np.ndarray[float]:
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
        A : Callable[[np.typing.ArrayLike], np.typing.ArrayLike]
            The classical driving term. Should be a vectorisable function of t.
            By default, is a sinusoidal function.

        Returns:
        --------
        ndarray[float] 
            The value of dc/dt at some time t.
        """

        if A is None:
            A = self.__SinusoidalDrivingTerm

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

    def Solve(self, tauAxis: np.ndarray[float], initialConditions: np.ndarray[complex], numT: int, steadyStateCutoff: float=25, debug: bool=False) -> CorrelationData:
        r"""
        Solves the system of ODEs for the expectations of our two-level system operators $(\langle \sigma_-(t) \rangle,\, \langle \sigma_+(t) \rangle,\, \langle \sigma_z(t) \rangle)$, and the double-time correlation functions $\langle \sigma_i (t) \sigma_j (t + \tau) \rangle$.
            
        Parameters
        ----------
        tauAxis : ndarray[float]
            The offsets from the initial time $t$ that the functions should be calculated on, in units of $\gamma_-^{-1}$.
       initialConditions : ndarray[complex]
            The initial conditions of the single-time correlation functions. This is used to calculate
            the initial conditions of the double-time correlations.
        numT : int
            The number of points within a steady-state period to use for the double-time correlation initial conditions.
        steadyStateCutoff : float
            The time, in units of $\gamma_-^{-1}$, after which we consider the system in steady-state. This is used
            to calculate the initial conditions for the double-time correlations.
        debug : bool
            Whether to print out debug progress statements.

        Returns
        -------
        CorrelationData
            A CorrelationData instance that contains all of the relevant information about the correlations.
        """

        # Ensures initial conditions are a numpy array.
        initialConditions = np.array(initialConditions, dtype=complex)

        self.__correlationData = CorrelationData(
            tauAxisDim = tauAxis,
            tauAxisSec = tauAxis / self.__params.decayConstant
        )

        # Stores the function parameters, since they are mostly the same for the single- and double-time
        # correlations.
        odeParams = {
            'fun' : self.__ClassicallyDrivenSSHEquations,
            't_span' : np.array([np.min(self.__correlationData.tauAxisSec), np.max(self.__correlationData.tauAxisSec)]),
            't_eval' : self.__correlationData.tauAxisSec,
            'rtol' : 1e-10,
            'atol' : 1e-12,
        }

        self.__correlationData.singleTime = self.__CalculateSingleTimeCorrelations(initialConditions, odeParams)

        # Calculates the single-time fourier expansions.
        numPeriods = 10
        dimPeriod = self.__params.decayConstant / self.__params.drivingFreq
        steadyStateMask = (steadyStateCutoff <= self.__correlationData.tauAxisDim) & (self.__correlationData.tauAxisDim <= steadyStateCutoff + dimPeriod)
        for i in range(3):
            self.__correlationData.singleTimeFourier.append(
                Fourier(self.__params.drivingFreq,
                        samples = self.__correlationData.singleTime[i][steadyStateMask],
                        samplesX = self.__correlationData.tauAxisSec[steadyStateMask],
                        numPeriods = numPeriods)
            )

        self.__correlationData.doubleTime = self.__CalculateDoubleTimeCorrelations(steadyStateCutoff, numT, odeParams)
  
        return self.__correlationData
    
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
        return integrate.solve_ivp(
            y0 = initialConditions,
            args = -self.__params.decayConstant,
            **odeParams
        ).y
    
    def __CalculateDoubleTimeCorrelations(self, steadyStateCutoff: int, numT: int, odeParams: dict, debug: bool=False) -> np.ndarray[complex]:
        r"""
        Calculates the double-time correlation functions.

        Parameters
        ----------
        steadyStateCutoff : int
            The point in time, in units if $\gamma_-^{-1}$, after which we consider the system to be in the steady state.
        numT : int
            The number of points within a steady state period to use as the initial conditions for a set of
            double-time correlation functions.
        odeParams: dict
            A dictionary containing the relevant parameters for the function which solves the ODE.
        debug : bool
            Whether to print out debug progress statements.
        """

        # Calculates the points within the steady state period that we want to use. The steady state axis
        # covers one period in the steady state, and is in seconds.
        startPoint = steadyStateCutoff / self.decayConstant
        endPoint = steadyStateCutoff / self.decayConstant + 1 / self.drivingFreq
        self.__correlationData.tAxisSec = np.linspace(startPoint, endPoint, numT)
        self.__correlationData.tAxisDim = self.__correlationData.tAxisSec * self.__params.decayConstant

        # Defines the double time solutions. The first dimension corresponds to the left-hand operator,
        # the second corresponds to the right hand operator, the third dimension corresponds to
        # the different times within a steady-state period that we consider our initial conditions at, and
        # the fourth dimension corresponds to the value of our time offset $\tau$.
        self._doubleTimeSolution = np.zeros((3, 3, self._tAxis.size, self._tauAxis.size), dtype=complex)

        # Calculates the double-time initial conditions based on the single-time correlations for
        # each time within the steady-state period that we want to calculate.
        doubleTimeInitialConditions = np.array([
            # When left-multiplying by $\sigma_-(t)$
            [
                np.zeros(self.__params.tAxisSec.size),
                -0.5 * (self.__correlationData.singleTimeFourier[2].Evaluate(self.__params.tAxisSec) - 1),
                self.__correlationData.singleTimeFourier[0].Evaluate(self.__params.tAxisSec)
            ],
            # When left-multiplying by $\sigma_+(t)$
            [
                0.5 * (self.__correlationData.singleTimeFourier[2].Evaluate(self.__params.tAxisSec) + 1),
                np.zeros(self.__params.tAxisSec.size),
                -self.__correlationData.singleTimeFourier[1].Evaluate(self.__params.tAxisSec)
            ],
            # When left-multiplying by $\sigma_z(t)$
            [
                -self.__correlationData.singleTimeFourier[0].Evaluate(self.__params.tAxisSec),
                self.__correlationData.singleTimeFourier[1].Evaluate(self.__params.tAxisSec),
                np.ones(self.__params.tAxisSec.size),
            ]], dtype=complex
        )

        iterable = enumerate(self._tAxis)
        if debug:
            print(f"Calculating double-time correlations for k = {self.k / np.pi:.2f}pi...")
            iterable = tqdm(iterable)

        # Loops through each initial condition time t.
        for tIndex, t in iterable:
            # Loops through all 3 operators that we can left-multiply by.
            for i in range(3):
                # Calculates the new initial conditions and inhomogenous term.
                newInitialCondition = -self.__params.decayConstant * self.__correlationData.__singleTimeFourier[i].Evaluate(t)[0]
                # Solves system.
                self._doubleTimeSolution[i, :, tIndex, :] = integrate.solve_ivp(
                    y0 = doubleTimeInitialConditions[i, :, tIndex],
                    args = (newInitialCondition,),
                    **odeParams
                ).y
        
        if debug:
            print('\n')
        
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
        
        self.__currentData = CurrentData()

        # Defines useful terms.
        Ek = self.t1 + self.t2 * np.exp(1j * self.k)
        phiK = np.angle(Ek)
        drivingSamples = self.__SinusoidalDrivingTerm(self.__params.tauAxisSec)

        # Calculates the current operator in terms of the previously calculated expectation values.
        self.__currentData.timeDomainData = self.t2 * (
            -np.sin(self.k - phiK - drivingSamples) * self._singleTimeSolution[2]
            + 1j * np.cos(self.k - phiK - drivingSamples) * (self._singleTimeSolution[1] - self._singleTimeSolution[0])
        )

        # Only considers the system in steady state for the Fourier transform, if desired.
        if steadyStateCutoff != None:
            mask = self.__params.tauAxisSec >= steadyStateCutoff / self.decayConstant
        else:
            mask = np.full(self.__params.tauAxisSec.size, True, dtype=bool)

        # The full axis which we have declared to be in a steady state.
        fullSteadyStateAxis = self.__params.tauAxisSec[mask]

        # Calculates the Fourier transform of the solution.
        self.__currentData.freqDomainData = np.fft.fftshift(np.fft.fft(self._currentTime[mask]))

        # Calculates the relevant frequency axis.
        sampleSpacing = (np.max(fullSteadyStateAxis) - np.min(fullSteadyStateAxis)) / fullSteadyStateAxis.size
        self.__currentData.freqAxis = np.fft.fftshift(np.fft.fftfreq(fullSteadyStateAxis.size, sampleSpacing))

        # Calculates the fourier expansions of the current data.
        self.__currentData.CalculateFourier(self.k, self.__params, self.__correlationData)

        return self.__currentData 