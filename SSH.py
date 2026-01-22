import numpy as np
import scipy.integrate as integrate
from typing import Callable, Any
import time

class SSH:
    r"""
    Models the 1-dimensional SSH model in momentum-space, with some classical driving term.

    Attributes
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
    tAxis : float
        The points in the time-domain that the single-time correlation functions are evaluated at.
    drivingTerm : Callable[[float], float]
        The chosen classical driving term for this system.
        By default, this is the classical sinusoidal driving term with amplitude $A_0$ and
        frequency $\Omega$. 
    initialConditions : np.ndarray[complex]
        The initial conditions of the system for $\tilde \sigma_-$, $\tilde \sigma_+$, and $\tilde \sigma_z$. It is important
        that the initial conditions are in the eigenbasis.
    numericalSol : Any
        The object returned from scipy.integrate.solve_ivp when we solve the
        ODE for the single-time correlations in the eigenbasis.
    """
    
    def __init__(self, k: float, t1: float, t2: complex, decayConstant: float, drivingAmplitude: float, drivingFreq: float):
        """
        Constructor for the class.

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

    def ClassicalDrivingTerm(self, t: np.typing.ArrayLike) -> np.typing.ArrayLike:
        r"""
        The classical sinusoidal driving term that we are considering. This models a laser with frequency $\Omega$ and amplitude $A_0$.

        Parameters
        ----------
            t : ArrayLike
                The absolute value of time.

        Returns
        -------
            ArrayLike
                The value of the driving term at that time.
        """

        return self.drivingAmplitude * np.sin(self.drivingFreq * t)

    def ClassicallyDrivenSSHEquations(self, t: float, c: np.ndarray[float], A: Callable[[np.typing.ArrayLike], np.typing.ArrayLike]) -> np.ndarray[float]:
        r"""
        The ODE (equations of motion) for the single-time expectations of $\sigma_-(t)$, $\sigma_+(t)$, and $\sigma_z(t)$. 

        Parameters
        ----------
            t : float
                The time t.
            c : ndarray[float, dtype[Any]]
                The 3-component single-time correlation array.
            A : Callable[[np.typing.ArrayLike], np.typing.ArrayLike]
                The classical driving term. Should be a vectorisable function of t.

        Returns:
        --------
            ndarraypfloat, dtype[Any]] 
                The value of dc/dt at some time t.
        """

        # Coefficient matrix at time t.
        Ek = self.t1 + self.t2 * np.exp(1j * self.k)
        phiK = np.angle(Ek)
        vZ = 2 * self.t2 * np.sin(self.k - phiK - 0.5 * A(t)) * np.sin(0.5 * A(t))
        vPm = 2j * self.t2 * np.cos(self.k - phiK - 0.5 * A(t)) * np.sin(0.5 * A(t))

        B = np.array([[2j * (vZ - np.abs(Ek)) - 0.5 * self.decayConstant  , 0                                                  ,  -1j * vPm            ],
                      [0                                                  , 2j * (np.abs(Ek) - vZ) - 0.5 * self.decayConstant  ,  -1j * vPm            ],
                      [2j * vPm                                           , 2j * vPm                                           ,  -self.decayConstant  ]], dtype=complex)
    
        # Inhomogenous part.
        d = np.array([0, 0, -self.decayConstant], dtype=complex)
    
        return B @ c + d

    def CalculateSingleTimeCorrelations(self, tAxis: np.ndarray[float],
                                        initialConditions: np.ndarray[complex],
                                        drivingTerm: Callable[[float], float]=ClassicallyDrivenSSHEquations,
                                        debug: bool=False) -> Any:
        r"""
        Solves the system ODE to get the single time correlations,.
    
        Parameters
        ----------
        tAxis : ndarray[float]
            The points at which to evaluate the solution to the ODE, in units of $\gamma_-^{-1}$.
        initialConditions : ndarray[complex]
            The initial conditions for the single time correlations.
        drivingTerm : Callable[[float], float]
            The function that describes the driving term, as a function of time in units of seconds.
        debug : bool
            Whether to print out debug statements.

        Returns
        -------
        Any
            The bunch object defined by scipy.integrate.solve_ivp.
        """

        # Note, the tDomain is defined in terms of $1 / \gamma_-$.
        self.tAxis = tAxis
        self.drivingTerm = drivingTerm

        if debug:
            print("Solving ODE...")
            startTime = time.perf_counter()

        self.numericalSol = integrate.solve_ivp(fun=self.ClassicallyDrivenSSHEquations,
                                        t_span= np.array([np.min(tAxis), np.max(tAxis)]) / self.decayConstant,
                                        y0=initialConditions,
                                        t_eval=tAxis / self.decayConstant,
                                        rtol=1e-10,
                                        atol=1e-12,
                                        args=(drivingTerm,))
        
        if debug:
            print(f"ODE solved in {time.perf_counter() - startTime:.2f}s.\n")

        return self.numericalSol

    def CalculateCurrentOperator(self, debug: bool=False) -> tuple[np.ndarray[float], np.ndarray[float]]:
        """
        Calculates the value of the current operator in the time domain.

        Parameters
        ----------
            debug : bool
                Whether to print out the debug statements.

        Returns
        -------
            ndarray[float]
                The first element of the tuple is the current operator in the time-domain, evaluated at the
                points along tAxis.
            ndarray[float]
                The second element is the fourier transform of the current operator, in the frequency domain.
        """

        if debug:
            print("Calculating current operator...")
            startTime = time.perf_counter()

        # Calculates the current operator in terms of the pauli matrices in the eigenbasis.
        currentCoeff = 1j * self.t2 * np.exp(1j * (self.k - self.drivingTerm(self.tAxis / self.decayConstant)))
        currentOperatorSol = -(currentCoeff * self.numericalSol.y[0] + currentCoeff.conjugate() * self.numericalSol.y[1])

        if debug:
            print(f"Current operator calculated in {time.perf_counter() - startTime:.2f}s.\n")
            print("Taking fourier transform of current operator...")
            startTime = time.perf_counter()

        # Takes the fourier transform of the current operator.
        fourierCurrentOperator = np.fft.fftshift(np.fft.fft(currentOperatorSol))

        if debug:
            print(f"Fourier transform calculated in {time.perf_counter() - startTime:.2f}s.\n")

        return currentOperatorSol, fourierCurrentOperator

    def GetFrequencyAxis(self):
        """
        Calculates the frequency axis for the fourier transform of the current operator.
        """

        sampleSpacing = (np.max(self.tAxis) - np.min(self.tAxis)) / (self.tAxis.size * self.decayConstant)
        return np.fft.fftshift(np.fft.fftfreq(self.tAxis.size, sampleSpacing))
    
    def TransformToCoordinateBasis(self, c: np.ndarray[float]) -> np.ndarray[float]:
        """
        Transforms the correlation functions in the eigenbasis to the coordinate basis.

        Parameters
        ----------
        c : np.ndarray[float]
            An array of shape (t_samples, 3), where the first dimension corresponds
            to the points in time that the correlation functions are evaluated at,
            and the second dimension corresponds to which of the single-time correlation functions
            we are considering.

        Returns
        -------
        np.ndarray[float]
            An array of the same shape as c, but where the vectors in the last
            dimension have been transformed to the coordinate basis.
        """

        Ek = self.t1 + self.t2 * np.exp(1j * self.k)
        phiK = np.angle(Ek)
        # Defines the common exponential terms
        exp = np.exp(1j * phiK)
        expConj = exp.conjugate()

        transformationMatrix = 0.5 * np.array([[-expConj, expConj, expConj ],
                                               [exp     , -exp   , exp     ],
                                               [2       , 2      , 0       ]], dtype=complex)
        
        return np.matvec(transformationMatrix, c)
    
    def TransformToEigenbasis(self, c: np.ndarray[float]) -> np.ndarray[float]:
        """
        Transforms the correlation functions in the coordinate to the eigenbasis.

        Parameters
        ----------
        c : np.ndarray[float]
            An array of shape (t_samples, 3), where the first dimension corresponds
            to the points in time that the correlation functions are evaluated at,
            and the second dimension corresponds to which of the single-time correlation functions
            we are considering.

        Returns
        -------
        np.ndarray[float]
            An array of the same shape as c, but where the vectors in the last
            dimension have been transformed to the eigenbasis.
        """

        Ek = self.t1 + self.t2 * np.exp(1j * self.k)
        phiK = np.angle(Ek)
        # Defines the common exponential terms
        exp = np.exp(1j * phiK)
        expConj = exp.conjugate()

        transformationMatrix = 0.5 * np.array([[-exp, expConj , 1 ],
                                               [exp , -expConj, 1 ],
                                               [exp , expConj , 0 ]], dtype=complex)
        
        return np.matvec(transformationMatrix, c)