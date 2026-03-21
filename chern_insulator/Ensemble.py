import numpy as np
from functools import cached_property
import matplotlib.pyplot as plt

from data import ModelParameters, EnsembleParameters, AxisData, CurrentData
from Model import Model

class Ensemble:
    """Runs a collection of model instances and finds the overall results of the model."""

    def __init__(self, params: EnsembleParameters) -> None:
        """
        Creates an ensemble instance.
        
        Parameters
        ----------
        params : EnsembleParameters
            The parameters for the ensemble.
        """

        self.__params = params

        # Stores the models in a dictionary.
        self.__models: dict[tuple[float, float], Model] = {}
        # The axis data shared by the system.
        self.__axes = None

    def AddMomentum(self, kValues: tuple[float, float] | list[tuple[float, float]] | np.ndarray[float]) -> None:
        """
        Adds one or more momentum points to the simulation.
        
        Parameters
        ----------
        kValues: tuple[float, float] | list[tuple[float, float]] | np.ndarray[float]
            The momentum points to simulate. Can be given as a single tuple containing
            (kx, ky), a list of tuples of that form, or a numpy array of shape (..., 2)
            where [:, 0] gives all of the kx values and [:, 1] gives all of the ky values.

            If inputting a single momentum value, must input as a tuple or a list with a single
            tuple in it - the shape of the numpy array becomes broken if we only give one kx, ky pair.
        """

        # If input is a tuple, make it a list of tuples.
        if isinstance(kValues, tuple):
            kValues = [kValues]

        # Now, iterating through kValues will return either tuples, or a 2 element
        # numpy array since we will by default iterate over the first dimension.
        # Either way, the following iteration code works. 
        for k in kValues:
            # Create the model parameters from the ensemble parameters.
            modelParams = ModelParameters.FromEnsemble(
                kx = k[0],
                ky = k[1],
                params = self.__params
            )

            # Stores the model in the dictionary with its momentum
            # tuple as the key.
            self.__models[(k[0], k[1])] = Model(modelParams)

    def Run(self, tauMax: float) -> None:
        """
        Runs all of the models.

        Parameters
        ----------
        tauMax : float
            The maximum non-dimensional time the system will solve for.
        """

        self.__axes = self.__CreateAxes(tauMax)

        for model in self.__models.values():
            model.Run(self.__axes)

    def SampleBrillouinZone(self, numK: int) -> None:
        """
        Samples the Brillouin Zone (kx, ky in [-pi, pi]) evenly on
        both axes to obtain a number of samples closest to numK.
        i.e. it will samples floor(sqrt(numK)) points on the x and y axes
        of the Brillouin zone.

        This function automatically adds the models with the associated
        momentum values to the ensemble.
        
        Parameters
        ----------
        numK : int
            The number of desired momentum points that we want to sample.
        """

        # Samples the Brillouin zone.
        sqrtK = np.floor(np.sqrt(numK)).astype(int)
        # Makes sure we have an even number of points along each axis.
        if sqrtK % 2 != 0:
            sqrtK += 1

        offset = 0.01
        axisPoints = np.linspace(-np.pi + offset, np.pi - offset, sqrtK)
        x, y = np.meshgrid(axisPoints, axisPoints)
 
        # Stacks x and y so that the last axis differentiates between them.
        momentums = np.stack((x.flatten(), y.flatten()), axis=-1)

        # Adds the momentum points to the ensemble.
        self.AddMomentum(momentums)

    def __CreateAxes(self, tauMax: float) -> AxisData:
        """
        Creates the axis data used for each model.
        
        Parameters
        ----------
        tauMax : float
            The maximum non-dimensional time the system will solve for.

        Returns
        -------
        AxisData:
            The object containing the axis data.
        """

        tauAxisDim = np.linspace(0, tauMax, 4000)
        tauAxisSec = tauAxisDim / self.__params.decayConstant

        sampleSpacing = (np.max(tauAxisSec) - np.min(tauAxisSec)) / tauAxisSec.size
        freqAxis = np.fft.fftshift(np.fft.fftfreq(tauAxisSec.size, sampleSpacing)) / self.__params.drivingFreq

        return AxisData(
            tauAxisDim = tauAxisDim,
            tauAxisSec = tauAxisSec,
            freqAxis = freqAxis
        )
    
    @cached_property
    def totalCurrent(self) -> CurrentData:
        return np.sum([model.currentData for model in self.__models.values()])
    
    @property
    def axes(self) -> AxisData:
        return self.__axes
    
    @property
    def params(self) -> EnsembleParameters:
        return self.__params