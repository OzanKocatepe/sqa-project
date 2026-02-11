import numpy as np
import matplotlib.pyplot as plt
import os

from .SSHSimulation import SSHSimulation

class SSHVisualiser:
    """Manages the plotting logic for SSHSimulation."""
 
    def __init__(self, sim: SSHSimulation, plotFolder: str):
        """Constructs an instance of the visualiser.
        
        Parameters
        ----------
        sim : SSHSimulation
            The instance of SSHSimulation which stores all of the information that we
            may want to plot. Should have already had Run() called, since that relies on
            other user inputs.
        plotFolder : str
            The location of the folder to save the plots in.
        """

        self.__sim = sim
        self.__plotFolder = plotFolder
        os.makedirs(self.__plotFolder, exist_ok=True)

        self.__axes = self.__sim.axes
        self.__plottingFunctions = [lambda z: np.abs(z), lambda z: z.real, lambda z: z.imag]
        self.__plottingPrefixes = ["Magnitude of", "Real Part of", "Imaginary Part of"]
        self.__operators = [r"\sigma_-", r"\sigma_+", r"\sigma_z"]
        self.__operatorSubscripts = ['-', '+', 'z']
        self.__tLabel = r"$t \gamma_-$"
        self.__tauLabel = r"$\tau \gamma_-$"

        plt.style.use('stylesheet.mplstyle')

    def PlotSingleTimeCorrelations(self, k: float, saveFigs: bool=False, show: bool=True, overplotFourierSeries: bool=False, overplotTAxis: bool=False) -> None:
        r"""Plots the single-time correlations $\langle \sigma_i (t) \rangle$
        
        Parameters
        ----------
        k : float
            The momentum to plot the single time correlations for.
        saveFigs : bool
            Whether to save the figures.
        show : bool
            Whether to show the figures.
        overplotFourierSeries : bool
            Whether to overplot the Fourier series for the single-time correlations.
        overplotTAxis : bool
            Whether to overplot the points along the axis that we have chosen as the steady-state
            initial conditions for the double-time correlations. Should all be within one steady-state period.
        """

        correlationData = self.__sim[k].correlationData

        nrows, ncols = 3, 3
        fig, ax = plt.subplots(nrows, ncols)
        plt.suptitle(self.__GenerateTitle(k))

        for row in range(nrows):
            for col in range(ncols):
                # Plots the single-time correlation.
                ax[row, col].plot(self.__axes.tauAxisDim,
                                  self.__plottingFunctions[col](correlationData.singleTime[row]),
                                  color='black')

                # Overplots the Fourier series.
                if overplotFourierSeries:
                    ax[row, col].plot(self.__axes.tauAxisDim,
                                      self.__plottingFunctions[col](correlationData.singleFourierSeries[row].Evaluate(self.__axes.tauAxisSec)),
                                      color='black')
                    
                # Overplots the t-axis.
                if overplotTAxis: 
                    for t in self.__axes.tAxisDim:
                        ax[row, col].axvline(t, color='red', linestyle='dashed')

                ax[row, col].set_xlabel(self.__tLabel)
                ax[row, col].set_ylabel(fr"{self.__plottingSuffixes[col]} $\langle {self.__operators[row]} \rangle$")

        if saveFigs:
            plt.savefig(f"{self.__plotFolder}/Single-Time Correlations.png", dpi=300)
        if show:
            plt.show()

    def PlotDoubleTimeCorrelations(self, k: float, format: str='noise', saveFigs: bool=False, show: bool=True, slice: list[tuple[int]]=None) -> None:
        r"""
        Plots the double-time correlations.
        
        Parameters
        ----------
        k : float
            The momentum to plot the single time correlations for.
        format : str
            'noise' if you want to plot the double-time correlations.
            'product' if you want to plot the product of the single-time correlations at two times.
            'connected' if you want to plot the connected correlators.
        saveFigs : bool
            Whether to save the figures.
        show : bool
            Whether to show the figures.
        slice : list[tuple[int]]
            A list of slices, corresponding to operators (i, j), specifying which double-time correlations to plot.
            If None, plots all 9.

        Raises
        ------
        ValueError
            If the given format isn't 'noise', 'product', or 'connected'.
        """

        correlationData = self.__sim[k].correlationData

        # If None, forms the iterable ((0, 0), (0, 1), ..., (2, 1), (2, 2)) to
        # loop through all 9 double-time correlations.
        if slice is None:
            i, j = np.meshgrid(range(3), range(3))
            iterable = zip((i.flatten(), j.flatten()))
        # Otherwise, only plots the specified ones.
        else:
            iterable = slice

        # Loops through every correlation.
        for i, j in iterable:
            # Sets which set of double-time data we are looking at.
            format = format.lower()
            if format == 'noise':
                operatorName = rf"$\langle {self.__operators[i]}(t) {self.__operators[j]}(t + \tau) \rangle$"
                subFolder = "Double-Time Correlations"
                data = correlationData.doubleTime

            else:
                # Calculates the product data, since that is needed for both other options.
                # Creates an array of shape (tAxis.size) where the values are $\sigma_i(t)$.
                firstTerm = correlationData.singleFourierSeries[i].Evaluate(self.__axes.tAxisSec)
                # Creates an array of shape (tAxis, tauAxis) where the values are t + tau.
                secondTerm = correlationData.singleFourierSeries[j].Evaluate(np.sum.outer(self.__axes.tAxisSec, self.__axes.tauAxisSec))
                # Creates a new imaginary axis on the first term to match the tau axis of the second term, and multiplies the two.
                productData = firstTerm[:, np.newAxis] * secondTerm

                if format == 'product':
                    operatorName = rf"$\langle {self.__operators[i]}(t) \rangle \langle {self.__operators[j]}(t + \tau) \rangle$"
                    subFolder = "Double-Time Products"
                    data = productData

                elif format == 'connected':
                    operatorName = fr"$\langle {self.__operators[i]}(t) {self.__operators[j]}(t + \tau) \rangle - \langle {self.__operators[i]}(t) \rangle \langle {self.__operators[j]}(t + \tau) \rangle$"
                    subFolder = "Double-Time Connected Correlators"
                    data = correlationData.doubleTime - productData
                
                else:
                    raise ValueError(f"The value of format must be 'noise', 'product', or 'connected', not '{format}'.")

            # Sets the title of the plot.
            title = fr"{operatorName} -- " + self.__GenerateTitle(k)
            plt.suptitle(title)

            nrows, ncols = 1, 2
            fig, ax = plt.subplots(nrows, ncols)

            for col in range(ncols):
                ax[col].pcolormesh(self.__axes.tauAxisDim,
                                   self.__axes.tAxisDim,
                                   self.__plottingFunctions[col + 1](data),
                                   cmap = 'bwr',
                                   shading = 'nearest')

                ax[col].set_title(f"{self.__plottingPrefixes[col]} Correlation")
                ax[col].set_xlabel(self.__tauLabel)
                ax[col].set_ylable(self.__tLabel)

            if saveFigs:
                os.makedirs(f"{self.__plotFolder}/{subFolder}", exist_ok=True)
                plt.savefig(f"{self.__plotFolder}/{subFolder}/{self.__operatorSubscripts[i], self.__operatorSubscripts[j]}.png", dpi=300)
            if show:
                plt.show()

    def __GenerateTitle(self, k: float | np.ndarray[float] | None) -> str:
        """Generates the title of a plot based on the parameters and momentum values displayed.
        
        Parameters:
        -----------
        k : float | ndarray[float] | None
            The momentum values to put in the title. Can be a scalar or a vector. Will be
            omitted from the title if None.
        """

        title = fr"$t_1 = {self._sim.params.t1},\, t_2 = {self._sim.params.t2},\, A_0 = {self._sim.params.drivingAmplitude},\, \Omega \approx {self._sim.params.drivingFreq:.3f},\, \gamma_- = {self._sim.params.decayConstant}$"

        # If there is no momentum, just return that title.
        if k is None:
            return title
        
        # Else, put in the k value(s).
        else:
            k = np.atleast_1d(k)
            
            # Checks if there are 3 or less k values.
            if k.size <= 3:
                kStr = list(k / np.pi)
                kStr = [f"{j}$\pi$" for j in kStr]
                kStr = "{" + kStr + "}"
            # Otherwise, assumes they're approximately evenly spaced and
            # writes them in the format of an np.linspace.
            else:
                kStr = f"[{np.min(k) / np.pi}$\pi$, {np.max(k) / np.pi}$\pi$, {k.size}]"

            # Appends this to the beginning of the title string.
            return kStr + " -- " + title
            

    def __PlotComplexDataInRows(self, x: np.ndarray[float], y: np.ndarray[complex], plotMagnitude: bool=False) -> None:
        """
        Plots each item of data in a row, with the columns corresponding to the components of the data.
        
        Parameters
        ----------
        x : ndarray[float]
            The x-values of the data, with the same shape as specified for y.
        y: ndarray[complex]
            The complex data to plot, of shape (nrows, numPoints) where the first dimension specifies the piece of data,
            and the second specifies the point along the x-axis.
        plotMagnitude : bool
            Whether to also plot the magnitude of the complex data in the first row.
        """

        nrows = 3 if plotMagnitude else 2
        ncols = y.shape[0]
        fig, ax = plt.subplots(nrows, ncols)

        for row in range(nrows):
            for col in range(ncols):
                # Controls whether to plot the magnitude or not.
                functionIndex = col
                if plotMagnitude:
                    functionIndex += 1

                # Plots the single-time correlation.
                ax[row, col].plot(x[row], self.__plottingFunctions[functionIndex](y[row]), color='black')