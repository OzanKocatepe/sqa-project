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

        # Stored in public variables so that they can be accessed again.
        self.t1 = t1
        self.t2 = t2
        self.decayConstant = decayConstant
        self.drivingAmplitude = drivingAmplitude
        self.drivingFreq = drivingFreq

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
        self._tAxis = None

    @property
    def tAxis(self) -> np.ndarray[float]:
        r"""Returns the tAxis.
        
        Returns
        -------
        ndarray[float]
            The points along which the solutions are evaluated in the time domain, in
            units of $\gamma_-^{-1}$.
        """

        if self._tAxis is None:
            raise ValueError("Call Run() first.")
        else:
            return self._tAxis

    @property
    def freqAxis(self) -> np.ndarray[float]: 
        """Gets the frequency axis.
        
        Returns
        -------
        ndarray[float]
            The frequencies that correspond to the amplitudes of the Fourier transform of the current operator
            found in CalculateCurrent(). Since the same tAxis is used for every model, this should be the same for every model,
            and so we just use the first model.
            
        Raises
        ------
        ValueError
            If Run() hasn't been called, or if no models have been added.
        """

        if not self._models:
            raise ValueError("No models have been added, so the frequency axis cannot be calculated.")
        else:
            try:
                return list(self._models.values())[0].freqAxis
            except (ValueError):
                raise ValueError("Call Run() first - there is currently no tAxis given.")

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
        r"""Adds one or more momentum points to the simulation.
        
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
        r"""Runs the simulations for all the momentum values.
        
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

        self._tAxis = tAxis

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

        current = np.array([model.currentTime for model in self._models.values()])
        fourier = np.array([model.currentFreq for model in self._models.values()])
        
        return np.sum(current, axis=0), np.sum(fourier, axis=0)