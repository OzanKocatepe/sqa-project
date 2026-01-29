import numpy as np
from .SSH import SSH
from .SSHParameters import SSHParameters
from .CurrentData import CurrentData

class SSHSimulation:
    """
    Controls the simulation of an arbitrary number of SSH models at once.
    All interactions with the SSH class should ideally occur through this class.
    """

    def __init__(self, params: SSHParameters):
        """
        Constructs an instance of SSHSimulation, setting the physical parameters of our system.

        Parameters
        ----------
        params : SSHParameters
            An instance of SSHParameters that contains the model parameters.
        """

        self.params = params
        
        # Dictionary of SSH instances.
        self.__models = {}

    def AddMomentum(self, k: list[float] | np.ndarray[float]) -> None:
        r"""Adds one or more momentum points to the simulation.
        
        Parameters
        ----------
        k : ndarray[float]
            The momentum points to simulate. Each point should be within the Brillouin Zone $[-\pi, \pi]$.
        """

        # Ensures our array is at least one dimensional.
        k = np.atleast_1d(k)

        for kPoint in k:
            self.__models[kPoint] = SSH(kPoint, self.params)

    def Run(self, tauAxis: np.ndarray[float], initialConditions: np.ndarray[complex], numT: int=5, steadyStateCutoff: float=25, debug: bool=False):
        r"""Runs the simulations for all the momentum values.
        
        Parameters
        ----------
        tauAxis : ndarray[float]
            The points in time (in units of $\gamma_-^{-1}$) that the solutions will be evaluated at.
        initialConditions : ndarray[complex]
            The initial conditions in the eigenbasis of the system.
        numT : int
            The number of t values within a steady state period to use.
        steadyStateCutoff : float
            The time (in units of $\gamma_-^{-1}$) which we consider the system to be in steady-state.
            i.e. we only consider the Fourier transform of the system after this point.
        debug : bool
            Whether to output debug progress statements.
        """

        iterable = self.__models.items()

        for k, model in iterable:
            if debug:
                print(f"Solving for momentum {k / np.pi:.2f}pi...")

            model.Solve(tauAxis, initialConditions, numT, debug=debug)
            model.CalculateCurrent(steadyStateCutoff)

            if debug:
                print('\n')

    def CalculateTotalCurrent(self) -> tuple[np.ndarray[complex], np.ndarray[complex]]:
        """Calculates the total current in the time and frequency domains.
        
        Returns
        -------
        np.ndarray[complex]
            The total current operator in the time domain.
        np.ndarray[complex]
            The fourier transform of the total current operator in the frequency domain.
        """

        current = np.array([model.currentData.timeDomainData for model in self.__models.values()])
        fourier = np.array([model.currentData.freqDomainData for model in self.__models.values()])

        current, fourier = np.sum(current, axis=0), np.sum(fourier, axis=0)

        return CurrentData(
            timeDomainData = current,
            freqDomainData = fourier,
            tauAxis = 
        )

    @property
    def momentums(self) -> np.ndarray[float]:
        """Returns the momentums currently within the simulation.
        
        Returns
        -------
        ndarray[float]
            The momentum values of the models currently stored within the simulation.
        """

        return np.array( list( self.__models.keys() ) )
    
    @property
    def models(self) -> dict[float, SSH]:
        """Returns the models currently in the simulation.
        
        Returns
        -------
        dict[float, SSH]
            A dictionary of SSH models, where the key is the momentum of the model.
        """

        return self.__models
    
    @property
    def tauAxisDim(self) -> np.ndarray[complex]:
        return list( self.__models.values() )[0].correlationData.tauAxisDim