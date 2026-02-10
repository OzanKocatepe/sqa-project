import numpy as np
import multiprocessing
from functools import cached_property
import os

from data import EnsembleParameters, ModelParameters, AxisData, CurrentData
from .SSH import SSH

class SSHSimulation:
    """
    Controls the simulation of an arbitrary number of SSH models at once.
    All interactions with the SSH class should ideally occur through this class.
    """

    def __init__(self, params: EnsembleParameters):
        """
        Constructs an instance of SSHSimulation, setting the physical parameters of our system.

        Parameters
        ----------
        params : EnsembleParameters
            An instance of EnsembleParameters that contains the model parameters.
        """

        self.__params = params
        
        # Dictionary of SSH instances.
        self.__models: dict[float, SSH] = {}
        # The axis information for our system, to be computed.
        self.__axes: AxisData = None
 
    def AddMomentum(self, kArr: list[float] | np.ndarray[float]) -> None:
        r"""Adds one or more momentum points to the simulation.
        
        Parameters
        ----------
        kArr : ndarray[float]
            The momentum points to simulate. Each point should be within the Brillouin Zone $[-\pi, \pi]$.
        """

        # Ensures our array is at least one dimensional.
        kArr = np.atleast_1d(kArr)

        for k in kArr:
            modelParams = ModelParameters.FromEnsemble(k, self.params)
            self.__models[k] = SSH(modelParams)
 
    def Run(self, initialConditions: np.ndarray[complex], tauAxisDim: np.ndarray[float], steadyStateCutoff: float, numT: int=10, numProcesses: int=None) -> None:
        r"""
        Runs the simulations for all the momentum values.
        
        Parameters
        ----------
        initialConditions : ndarray[complex]
            The initial conditions in the eigenbasis of the system.
        tauAxisDim : ndarray[float]
            The points in time (in units of $\gamma_-^{-1}$) that the solutions will be evaluated at.
        steadyStateCutoff : float
            The time (in units of $\gamma_-^{-1}$) which we consider the system to be in steady-state.
        numT : int
            The number of t values within a steady state period to use. If <2, will be set to 2
            so that the integration step for the double-time current correlations actually averages
            over a steady-state period.
        numProcesses : int
            The number of processes to run the code with. If None, it will use all available cores.
        """

        maxProcesses = np.max(1, os.cpu_count() - 1)

        # If None, uses all available cores except for 1 left for the system.
        if numProcesses is None:
            numProcesses = maxProcesses

        # Otherwise, filters the input.
        else:
            # Ensures there is at least one process.
            # Also deals with negative values.
            numProcesses = np.max(1, numProcesses)
            # Ensures numProcesses doesn't exceed the maximum number of allowed processes.
            numProcesses = np.min(numProcesses, maxProcesses)

        # Calculates the t- and tau- axes.
        self.__CalculateAxisData(tauAxisDim, steadyStateCutoff, numT)

        # Creates the arguments dispatched to each process.
        args = []
        for kIndex, dictItem in enumerate(self.__models.items()):
            k, model = dictItem
            # Appends a tuple of all the parameters for this process.
            args.append((
                k, kIndex, model, initialConditions
            ))

        # Runs the model on the desired number of cores.
        with multiprocessing.Pool(numProcesses) as p:
            newModels = p.map(func = self.__RunWrapper, iterable=args)

        # Since we pass a copy of our model to the processes, when they run the model
        # we need to store it in newModels and then replace each model with its new version.
        # Otherwise, the models which are actually run are immediately lost.
        for processIndex, processArgs in enumerate(args):
            k = processArgs[0]
            self.__models[k] = newModels[processIndex]

    def __RunWrapper(self, args: tuple) -> SSH:
        """
        The wrapper function that allows each model to be solved in parallel by different processes.

        Parameters
        ----------
        args : tuple
            A tuple containing the momentum, the index of the momentum in the list of
            momentums created in Run(), the current model and the initial conditions.
        """

        k, kIndex, model, initialConditions = args

        print(f"Solving for momentum {k / np.pi:.2f}pi ({kIndex + 1}/{self.numModels})...")
        model.SolveCorrelations(self.__axes, initialConditions)
        model.CalculateCurrent()

    def __CalculateAxisData(self, tauAxisDim: np.ndarray[float], steadyStateCutoff: float, numT: float) -> None:
        """
        Calculates the axis data to be used for all SSH models in the simulation.

        Parameters
        ----------
        tauAxisDim : ndarray[float]
            The points in time (in units of $\gamma_-^{-1}$) that the solutions will be evaluated at.
        steadyStateCutoff : float
            The time (in units of $\gamma_-^{-1}$) which we consider the system to be in steady-state.
        numT : int
            The number of t values within a steady state period to use. If <2, will be set to 2
            so that the integration step for the double-time current correlations actually averages
            over a steady-state period.
        """

        # Sets a minimum value of 2.
        numT = np.max(2, numT)

        # Finds the start and end points of a steady-state period in dimensionless unit,
        # assuming it has the same frequency as the driving frequency.
        dimensionlessPeriod = self.__params.decayConstant / self.__params.drivingFreq
        tAxisDim = np.linspace(steadyStateCutoff, steadyStateCutoff + dimensionlessPeriod, numT)

        # Calculates the frequency axis based on the steady state axis.
        mask = tauAxisDim >= steadyStateCutoff
        steadyStateAxis = self.__correlationData.tauAxisSec[mask]
        sampleSpacing = (np.max(steadyStateAxis) - np.min(steadyStateAxis)) / steadyStateAxis.size
        freqAxis = np.fft.fftshift(np.fft.fftfreq(steadyStateAxis.size, sampleSpacing))

        # Stores the data in the current instance's axis data.
        self.__axes = AxisData(
            tauAxisDim = tauAxisDim,
            tauAxisSec = tauAxisDim / self.__params.decayConstant,
            tAxisDim = tAxisDim,
            tAxisSec = tAxisDim / self.__params.decayConstant,
            freqAxis = freqAxis,
            steadyStateCutoff = steadyStateCutoff
        )
    
    @property
    def momentums(self) -> np.ndarray[float]:
        return np.array( list( self.__models.keys() ) )
    
    @property
    def models(self) -> list[SSH]:
        return list( self.__models.values() )

    @property
    def numModels(self) -> int:
        return len(list(self.__models.items()))
    
    @property
    def axes(self) -> AxisData:
        return self.__axes
    
    @cached_property
    def totalCurrent(self) -> CurrentData:
        currents = [model.currentData for model in self.models]
        return np.sum(currents)