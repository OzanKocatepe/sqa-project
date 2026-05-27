import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from config.paths import DATA_DIR
import os

from data import EnsembleParameters, ModelParameters, AxisData, CurrentData
from .model import Model
from . import current_solver

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
        self.__models: dict[tuple[float, float], Model | None] = {}
        # The axis data shared by the system.
        self.__axes = None
        # Stores the final current information.
        self.meanCurrent = None

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

    def Run(self, tauMax: float, numT: float, numProcesses: int | None=1) -> None:
        """
        Runs all of the models.

        Parameters
        ----------
        tauMax : float
            The maximum non-dimensional time the system will solve for.
        numT : float
            The number of points within one period of the driving to evaluate
            to use as the starting points for the double-time correlations.
        numProcesses : int | None, optional
            The number of processes to use when running the models. If None, will use all but
            one core (unless we only have 1 core, in which case we will use 1 core).
            Otherwise, input will be clamped between 1 and the number of cores.

            If 1 is given, multiprocessing won't be used and the models will just be run
            sequentially.

            WARNING: It is possible to manually choose to use all cores of the machine.
            This may cause unintended behaviour.
        """

        # Sanitizes the numProcesses input.
        if numProcesses is None:
            # Uses all but one core, unless we only have one core.
            numProcesses = np.max([1, mp.cpu_count() - 1])   
        else:
            # Clamps the number of processes between 1 and the number of cores. 
            # Note that it is possible to manually set the number of processes to
            # be equal to the number of cores.
            numProcesses = np.clip(numProcesses, 1, mp.cpu_count())

        # Limits the number of allowed processes to be less than or equal to the number of models we have.
        # No point in creating processes for models we don't have.
        numProcesses = np.clip(numProcesses, 1, len(self.__models))

        self.__axes = self.__CreateAxes(tauMax, numT)

        if numProcesses == 1:
            for key, model in tqdm(
                self.__models.items(),
                # disable = True,
                mininterval = 5,
                desc=f"Running models (Delta = {self.__params.delta})"
            ):
                model.Run(self.__axes)

                # Adds current to mean current.
                if self.meanCurrent is None:
                    self.meanCurrent = model.currentData
                else:
                    self.meanCurrent = self.meanCurrent + model.currentData

                # Deletes the model to free memory.
                self.__models[key] = None

        else:
            ctx = mp.get_context('fork')
            with ctx.Pool(processes=numProcesses) as pool:
                # Creates a generator for the tasks so not all models have to be built at once.
                tasks = (
                    (key, model, self.__axes)
                    for key, model in self.__models.items()
                )

                # Creates progress bar.
                pbar = tqdm(
                    total=len(self.__models),
                    desc=f"Running models (Delta = {self.__params.delta})",
                    # disable = True,
                    mininterval = 5,
                    position=0
                )

                # Evaluates models in parallel - returns values as they come in.
                for key, currentData in pool.imap(
                    self._MultiProcessingRun,
                    tasks,
                    # chunksize = len(self.__models) // (numProcesses * 8) + 1
                    chunksize = 1
                ):
                    # Adds the current to the total current.
                    if self.meanCurrent is None:
                        self.meanCurrent = currentData
                    else:
                        self.meanCurrent = self.meanCurrent + currentData

                    # Frees reference to the model to free memory.
                    self.__models[key] = None
                    # Increments progress bar manually, because larger batchsizes don't mess up
                    # the automatic incrementing.
                    pbar.update(1)

        self.meanCurrent = self.meanCurrent / len(self.__models)

        # Calculates some properties that require the mean current, since their non-linear.
        # Calculates 2n - 1 fourier coefficients since thats the most we will need to calculate g2(0)
        # for 1 to n.
        current_fourier_coefficients = current_solver.calculate_current_fourier_coefficients(
            self.__params,
            self.meanCurrent.total_current,
            self.__axes.tau_axis_sec,
            2 * self.__params.maxN - 1
        )

        self.meanCurrent.semiclassical_mode_population = current_solver.calculate_semiclassical_mode_population(
            self.__params,
            current_fourier_coefficients
        )

        self.meanCurrent.second_order_correlation_function = current_solver.calculate_second_order_correlation_function(
            self.__params,
            current_fourier_coefficients
        )
 
    def _MultiProcessingRun(self, args: tuple[tuple[float, float], Model, AxisData]) -> tuple[tuple[float, float], Model]:
        """
        Creates a process to run the model. Utilised by Run() to
        run the models in parallel.

        Parameters
        ----------
        args : tuple[tuple[float, float], Model, AxisData]
            Contains the arguments for the function, passed in as an argument.
            First element is the momentum tuple associated with the model. Required to
            store the results in the correct place in self.__models.
            Second element is the model instance to run.
            Third element is the axis data to run the model with. Required due to
            the fact that we cannot share self.__axes between processes. 

        Returns
        -------
        tuple[float, float]
            Returns the key so that we can store the results in the appropriate place in self.__models.
        CurrentData
            The current data calculated from this model.
        """

        key, model, axes = args
        
        model.Run(axes)
        return key, model.currentData

    def SampleBrillouinZone(self, numK: int) -> None:
        """
        Samples the Brillouin Zone (kx, ky in [-pi, pi]) using a
        numK-by-numK grid.

        This function automatically adds the models with the associated
        momentum values to the ensemble.
        
        Parameters
        ----------
        numK : int
            The side length of the grid we want to sample.
        """

        offsetX, offsetY = 0, 0
        # print(f"x-offset: {offsetX}, y-offset: {offsetY}")
        xPoints = np.linspace(-np.pi + offsetX, np.pi - offsetX, numK, endpoint=False) + np.pi / numK
        yPoints = np.linspace(-np.pi + offsetY, np.pi - offsetY, numK, endpoint=False) + np.pi / numK
        x, y = np.meshgrid(xPoints, yPoints)

        # plt.scatter(x / np.pi, y / np.pi, s=2, color='black')
        # plt.grid(True, which='both')
        # plt.axis('equal')
        # plt.show()

        # Masks our the (0, 0) point, since the code breaks there.
        # Better than enforcing that numK has to be even.
        zeroMask = (x == 0) & (y == 0)
        x = x[~zeroMask]
        y = y[~zeroMask]
         
        # Stacks x and y so that the last axis differentiates between them.
        momentums = np.stack((x.flatten(), y.flatten()), axis=-1)

        # Adds the momentum points to the ensemble.
        self.AddMomentum(momentums)

    def SaveCurrent(self) -> None:
        """Saves the current to disk.
        
        Saves the mean current object to the data/ folder. If this folder doesn't exist,
        it is created.
        """

        os.makedirs(DATA_DIR, exist_ok = True)
        file = DATA_DIR / f"D={self.__params.delta}, k={int(np.sqrt(len(self.__models)))}"
        np.save(file, (self.__axes, self.meanCurrent))

    def __CreateAxes(self, tauMax: float, numT: float) -> AxisData:
        """
        Creates the axis data used for each model.
        
        Parameters
        ----------
        tauMax : float
            The maximum non-dimensional time the system will solve for.
        numT : float
            The number of points within one period of the driving to evaluate
            to use as the starting points for the double-time correlations.

        Returns
        -------
        AxisData:
            The object containing the axis data.
        """

        tauAxisDim = np.linspace(0, tauMax, 4000)
        tauAxisSec = tauAxisDim / self.__params.decayConstant

        # The tAxis is numT points from time 0 to the first period of the driving.
        # This doesn't include the endpoint, since that is already considered by the point at time 0,
        # since the single-time correlations are periodic.
        tAxisSec = np.linspace(0, 1 / self.__params.drivingFreq, numT, endpoint=False)
        tAxisDim = tAxisSec / self.__params.decayConstant

        sampleSpacing = (np.max(tauAxisSec) - np.min(tauAxisSec)) / tauAxisSec.size
        freqAxis = np.fft.fftshift(np.fft.fftfreq(tauAxisSec.size, sampleSpacing)) / self.__params.drivingFreq

        return AxisData(
            tau_axis_dim = tauAxisDim,
            tau_axis_sec = tauAxisSec,
            t_axis_dim = tAxisDim,
            t_axis_sec = tAxisSec,
            freq_axis = freqAxis
        )
     
    @property
    def axes(self) -> AxisData:
        return self.__axes
    
    @property
    def params(self) -> EnsembleParameters:
        return self.__params
    
    @property
    def models(self) -> dict[tuple[float, float], Model]:
        return self.__models