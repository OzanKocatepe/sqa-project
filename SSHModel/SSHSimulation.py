import numpy as np
import multiprocessing
from functools import cached_property
import os
import pickle
import gzip

from .data import EnsembleParameters, ModelParameters, AxisData, CurrentData, CorrelationData
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

    def __getitem__(self, kArr: float | list[float] | np.ndarray[float]) -> SSH | list[SSH]:
        """
        Returns the model with the given momentum.
        
        Parameters
        ----------
        k : float | list[float] | np.ndarray[float]
            The momentum(s) of the models to extract.
            
        Returns
        -------
        SSH | list[SSH]
            The model(s) with the given momentum(s), if they exist. Returns an SSH object,
            or a list of SSH objects, depending on how many are returned.
        """

        # Makes the object iterable, even if its a scalar.
        kArr = np.atleast_1d(kArr)

        # Saves the models to a list.
        output = []
        for k in kArr:
            output.append(self.__models[k])

        # if there is only one model, return the model without the wrapping list.
        if len(output) == 1:
            return output[0]
        else:
            return output
 
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
            modelParams = ModelParameters.FromEnsemble(k, self.__params)
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

        maxProcesses = np.max((1, os.cpu_count() - 1))

        # If None, uses all available cores except for 1 left for the system.
        if numProcesses is None:
            numProcesses = maxProcesses

        # Otherwise, filters the input.
        else:
            # Ensures there is at least one process.
            # Also deals with nega(tive values.
            numProcesses = np.max((1, numProcesses))
            # Ensures numProcesses doesn't exceed the maximum number of allowed processes.
            numProcesses = np.min((numProcesses, maxProcesses))

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
            newModels = p.map(func = self._RunWrapper, iterable=args)

        # Since we pass a copy of our model to the processes, when they run the model
        # we need to store it in newModels and then replace each model with its new version.
        # Otherwise, the models which are actually run are immediately lost.
        for processIndex, processArgs in enumerate(args):
            k = processArgs[0]
            self.__models[k] = newModels[processIndex]

    def _RunWrapper(self, args: tuple) -> SSH:
        """
        The wrapper function that allows each model to be solved in parallel by different processes.
        Cannot be private due to the way multiprocess.map sends the function to the processes - left as
        protected as the next best option.

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
        return model

    def __CalculateAxisData(self, tauAxisDim: np.ndarray[float], steadyStateCutoff: float, numT: float) -> None:
        r"""
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
        numT = np.max((2, numT))

        # Finds the start and end points of a steady-state period in dimensionless unit,
        # assuming it has the same frequency as the driving frequency.
        dimensionlessPeriod = self.__params.decayConstant / self.__params.drivingFreq
        tAxisDim = np.linspace(steadyStateCutoff, steadyStateCutoff + dimensionlessPeriod, numT)

        # Calculates the frequency axis based on the steady state axis.
        mask = tauAxisDim >= steadyStateCutoff
        steadyStateAxis = tauAxisDim[mask] / self.__params.decayConstant
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

    def Save(self, fileDir: str) -> None:
        """
        Saves the current simulation instance into a pickle file.
        
        Parameters
        ----------
        fileDir : str
            The directory to save the file to.     
        """

        with gzip.open(f"{fileDir}/numK: {self.numModels}, numT: {self.__axes.tAxisDim.size}.pkl.gz", 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def Load(cls, filePath: str) -> SSHSimulation:
        """
        Loads an instance of SSHSimulation from a pickled file.
        
        Parameters
        ----------
        filePath : str
            Where to find the file to load.
            
        Returns
        -------
        SSHSimulation
            The loaded instance.
        """
        
        with gzip.open(filePath, 'rb') as file:
            obj = pickle.load(file)
        return obj
    
    def ExtractModels(self, kArr: float | list[float] | np.ndarray[float]) -> SSHSimulation:
        """
        Copies a particular subset of the models stored in this SSHSimulation instance into
        another instance.
        
        Parameters
        ----------
        kArr : float | list[float] | np.ndarray[float]
            The momentum of the models to extract.

        Returns
        -------
        SSHSimulation
            An SSHSimulation instance containing those models.
        """

        # Makes the momentum values iterable.
        kArr = np.atleast_1d(kArr)

        # Creates the dictionary of the models
        # we want to extract.
        extractedModels = {}
        for k in kArr:
            extractedModels[k] = self.__models[k]

        # Adds the models to a fresh SSHSimulation.
        newSimulation = SSHSimulation(self.__params)
        newSimulation.AppendModels(extractedModels)
        newSimulation.axes = self.__axes
        return newSimulation

    def AppendModels(self, models: dict[float, SSH]) -> None:
        """
        Appends models to the internal dictionary.
        Will override older models with the given models if they
        share a momentmum.
        
        Parameters
        ----------
        models : dict[float, SSH]
            A dictionary containing the SSH models.
        """

        self.__models |= models

    @property
    def momentums(self) -> np.ndarray[float]:
        return np.array( list( self.__models.keys() ) )
    
    @property
    def models(self) -> dict[float, SSH]:
        return self.__models

    @property
    def numModels(self) -> int:
        return len(list(self.__models.items()))
    
    @property
    def axes(self) -> AxisData:
        return self.__axes
    
    @axes.setter
    def axes(self, axes: AxisData) -> None:
        """
        Only allows setting the axes if they don't exist yet. Used only
        for when we are creating a new SSHSimulation system.
        """
        if self.__axes is None:
            self.__axes = axes

    @cached_property
    def totalCorrelations(self) -> CorrelationData:
        correlations = [model.correlationData for model in self.models]
        return np.sum(correlations)
    
    @cached_property
    def totalCurrent(self) -> CurrentData:
        currents = [model.currentData for model in self.models]
        return np.sum(currents)
    
    @property
    def params(self) -> EnsembleParameters:
        return self.__params