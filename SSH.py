import numpy as np
import scipy.integrate as integrate
from typing import Callable, Any

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
    def tAxis(self) -> np.ndarray[float]:
        """Gets the time axis.
        
        Returns
        -------
        ndarray[float]
            The points in time that we are evaluating the solutions at, in units of $\gamma^{-1}$.
            
        Raises
        ------
        ValueError
            If Solve() has not been called, meaning no time axis has been given to the system.
        """

        if self._tAxis == None:
            raise ValueError("Call Solve() first.")
        else:
            return self._tAxis * self.decayConstant
        
    @property
    def Solution(self) -> Any:
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
        if self._solution == None:
            raise ValueError("Call Solve() first.")
        else:
            return self._solution
        
    @property
    def currentTime(self):
        """Gets the current operator in the time domain.
        
        Returns
        -------
        ndarray[float]
            The values of the current operator at each time in the given tAxis.
            
        Raises
        ------
        ValueError
            If CalculateCurrent() has not been called, meaning the current operator hasn't been calculated yet.
        """

        if self._currentTime== None:
            raise ValueError("Call CalculateCurrent() first.")
        else:
            return self._currentTime

    @property
    def currentFreq(self):
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

        if self._currentFreq== None:
            raise ValueError("Call CalculateCurrent() first.")
        else:
            return self._currentFreq

    @property
    def freqAxis(self):
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

        if self._freqAxis == None:
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

        if drivingTerm == None:
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
        """Calculates the current operator for the given parameters.
        
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

        if self._solution == None:
            raise ValueError("Call Solve() first.")

        # Only considers the system in steady state, if desired.
        if steadyStateCutoff != None:
            mask = self._tAxis >= steadyStateCutoff / self.decayConstant
        else:
            mask = np.full(self._tAxis.size, True, dtype=bool)

        steadyStateAxis = self._tAxis[mask]
        steadyStateSolution = self._solution.y[:, mask]

        # Defines useful terms.
        Ek = self.t1 + self.t2 * np.exp(1j * self.k)
        phiK = np.angle(Ek)
        drivingSamples = self._drivingTerm(steadyStateAxis)

        # Calculates the current operator in terms of the previously calculated expectation values.
        self._currentTime = self.t2 * (
            -np.sin(self.k - phiK - drivingSamples) * steadyStateSolution[2]
            + 1j * np.cos(self.k - phiK - drivingSamples) * (steadyStateSolution[1] - steadyStateSolution[0])
        )

        # Calculates the Fourier transform of the solution.
        self._currentFreq = np.fft.fftshift(np.fft.fft(self._currentTime))

        # Calculates the relevant frequency axis.
        sampleSpacing = (np.max(steadyStateAxis) - np.min(steadyStateAxis)) / steadyStateAxis.size
        self._freqAxis = np.fft.fftshift(np.fft.fftfreq(steadyStateAxis.size, sampleSpacing))

        return self._currentTime, self._currentFreq