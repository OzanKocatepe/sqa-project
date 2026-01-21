import numpy as np
import scipy.integrate as integrate
from typing import Callable, Any

class SSH:
    
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
        """
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
        """
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

    def CalculateSingleTimeCorrelations(self, tAxis: np.ndarray[float], initialConditions: np.ndarray[complex], drivingTerm: Callable[[float], float]) -> Any:
        """
        Solves the system ODE to get the single time correlations,.
    
        Parameters
        ----------
        tAxis : ndarray[float]
            The points at which to evaluate the solution to the ODE, in units of $\gamma_-^{-1}$.
        initialConditions : ndarray[complex]
            The initial conditions for the single time correlations.
        drivingTerm : Callable[[float], float]
            The function that describes the driving term, as a function of time in units of seconds.

        Returns
        -------
        Any
            The bunch object defined by scipy.integrate.solve_ivp.
        """

        # Note, the tDomain is defined in terms of $1 / \gamma_-$.
        self.tAxis = tAxis
        self.tDomain = np.array([np.min(tAxis), np.max(tAxis)])
        self.n_tSamples = tAxis.size
        self.drivingTerm = drivingTerm

        self.numericalSol = integrate.solve_ivp(fun=self.ClassicallyDrivenSSHEquations,
                                        t_span=self.tDomain / self.decayConstant,
                                        y0=initialConditions,
                                        t_eval=tAxis / self.decayConstant,
                                        rtol=1e-10,
                                        atol=1e-12,
                                        args=(drivingTerm,))
        
        return self.numericalSol


    def CalculateCurrentOperator(self) -> tuple[np.ndarray[float], np.ndarray[float]]:
        """
        Calculates the value of the current operator in the time domain.

        Returns:
            ndarray[float]
                The first element of the tuple is the current operator in the time-domain, evaluated at the
                points along tAxis.
            ndarray[float]
                The second element is the fourier transform of the current operator, in the frequency domain.
        """

        # Calculates the current operator in terms of the pauli matrices in the eigenbasis.
        currentCoeff = 1j * self.t2 * np.exp(1j * (self.k - self.drivingTerm(self.tAxis / self.decayConstant)))
        currentOperatorSol = -(currentCoeff * self.numericalSol.y[0] + currentCoeff.conjugate() * self.numericalSol.y[1])

        # Takes the fourier transform of the current operator.
        fourierCurrentOperator = np.fft.fftshift(np.fft.fft(currentOperatorSol))

        return currentOperatorSol, fourierCurrentOperator

    def GetFrequencyAxis(self):
        """
        Calculates the frequency axis for the fourier transform of the current operator.
        """

        sampleSpacing = (self.tDomain[1] - self.tDomain[0]) / (self.n_tSamples * self.decayConstant)
        return np.fft.fftshift(np.fft.fftfreq(self.n_tSamples, sampleSpacing))