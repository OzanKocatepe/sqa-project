import numpy as np
from SSH import SSH
from typing import Callable

class SSHSimulation:
    """
    Controls the simulation of an arbitrary number of SSH models at once.
    All interactions with the SSH class should ideally occur through this class.
    """

    def __init__(self, t1: float, t2: float, decayConstant: float, drivingAmplitude: float, drivingFreq: float):
        """
        Constructs an instance of SSHSimulation, setting the physical parameters of our system.

        Parameters
        ----------
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

        # Sets the parameters of the system.
        self._params = {
            't1' : t1,
            't2' : t2,
            'decayConstant' : decayConstant,
            'drivingAmplitude' : drivingAmplitude,
            'drivingFreq' : drivingFreq
        }
        
        # Dictionary of SSH instances.
        self._models = {}

    @property
    def momentums(self) -> np.ndarray[float]:
        """Returns the momentums currently within the simulation.
        
        Returns
        -------
        ndarray[float]
            The momentum values of the models currently stored within the simulation.
        """

        return np.array( list( self._models.keys() ) )

    def AddMomentum(self, k: np.ndarray[float]):
        """Adds one or more momentum points to the simulation.
        
        Parameters
        ----------
        k : ndarray[float]
            The momentum points to simulate. Each point should be within the Brillouin Zone $[-\pi, \pi]$.
        """

        # Dealing with the one momentum case.
        if np.array(k).size == 1:
            k = [k]

        for kPoint in k:
            self._models[kPoint] = SSH(k = kPoint, **self._params)

    def Run(self, tAxis: np.ndarray[float], initialConditions: np.ndarray[complex], steadyStateCutoff: float=None, drivingTerm: Callable[[float], float]=None):
        """Runs the simulations for all the momentum values.
        
        Parameters
        ----------
        tAxis : ndarray[float]
            The points in time (in units of $\gamma_-^{-1}$) that the solutions will be evaluated at.
        initialConditions : ndarray[complex]
            The initial conditions in the eigenbasis of the system.
        steadyStateCutoff : float
            The time (in units of $\gamma_-^{-1}$) which we consider the system to be in steady-state.
            i.e. we only consider the Fourier transform of the system after this point.
        drivingTerm: Callable[[float], float]
            The classical driving term to use for the simulation.
            By default, this is a sinusoidal term.
        """

        for k, model in self._models.items():
            model.Solve(tAxis, initialConditions, drivingTerm)
            model.CalculateCurrent(steadyStateCutoff)

    def CalculateTotalCurrent(self) -> tuple[np.ndarray[complex], np.ndarray[complex]]:
        """Calculates the total current in the time and frequency domains.
        
        Returns
        -------
        np.ndarray[complex]
            The total current operator in the time domain.
        np.ndarray[complex]
            The fourier transform of the total current operator in the frequency domain.
        """

        current = [model.currentTime for model in self._models.values()]
        fourier = [model.currentFreq for model in self._models.values()]
        
        return np.sum(current), np.sum(fourier)