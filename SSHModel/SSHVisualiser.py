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
        self.__fLabel = r"$f / \Omega$"

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

    def __GenerateTitle(self, k: float | list[float] | np.ndarray[float] | None) -> str:
        """Generates the title of a plot based on the parameters and momentum values displayed.
        
        Parameters:
        -----------
        k : float | list[float] | ndarray[float] | None
            The momentum values to put in the title. Can be a scalar or a vector. Will be
            omitted from the title if None.

        Returns
        -------
        str:
            The title string.
        """

        title = fr"$t_1 = {self.__sim.params.t1},\, t_2 = {self.__sim.params.t2},\, A_0 = {self.__sim.params.drivingAmplitude},\, \Omega \approx {self.__sim.params.drivingFreq:.3f},\, \gamma_- = {self.__sim.params.decayConstant}$"

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
        
    def PlotCurrent(self, k: float | list[float] | np.ndarray[float] | None=None, saveFigs: bool=False, show: bool=True, overplotFourierSeries: bool=False) -> None:
        r"""
        Plots the expectation of the current operator, $\langle j(t) \rangle$, summed over the desired momentum values.
        
        Parameters
        ----------
        k : float | list[float] | ndarray[float]
            The momentum or momentums to plot the total current for.
        saveFigs : bool
            Whether to save the figures.
        show : bool
            Whether to show the figures.
        overplotFourierSeries : bool
            Whether to overplot the Fourier series for the single-time correlations.
        """

        if k is not None:
            extractedSimulation = self.__sim.ExtractModels(k)
            currentData = extractedSimulation.totalCurrent
        else:
            currentData = self.__sim.totalCurrent
            k = self.__sim.momentums

        self.__PlotOneComplexFunction(
            x = self.__axes.tauAxisDim,
            y = currentData.timeDomainCurrent,
            dataName = r"$\langle j(t) \rangle$",
            title = plt.suptitle(self.__GenerateTitle(k)),
            createFigure = True
        )

        if overplotFourierSeries:
            self.__PlotOneComplexFunction(
                x = self.__axes.tauAxisDim,
                y = currentData.currentFourierSeries.Evaluate(self.__axes.tauAxisDim),
                dataName = None,
                title = None,
                createFigure = False
            )

        if saveFigs:
            plt.savefig(f"{self.__plotFolder}/Time-Domain Current.png", dpi=300)
        if show:
            plt.show()

    def PlotIntegratedDoubleTimeCurrent(self, format: str='noise', saveFigs: bool=False, show: bool=True) -> None:
        r"""
        Plots the double-time current operator, integrated over a steady-state period w.r.t. $t$, so the data becomes a function of
        just $\tau$.

        Parameters:
        -----------
        format : str
            'noise' if you want to plot the double-time current correlation.
            'product' if you want to plot the product of the single-time current at two times.
            'connected' if you want to plot the connected current correlator.
        saveFigs : bool
            Whether to save the figures.
        show : bool
            Whether to show the figures.

        Raises
        ------
        ValueError
            If the given format isn't 'noise', 'product', or 'connected'.
        """

        currentData = self.__sim.totalCurrent

        format = format.lower()
        if format == 'noise':
            operatorName = r"$\int dt\, \langle j(t) j(t + \tau) \rangle$"
            data = currentData.integratedDoubleTimeCurrent
            fileName = "Double-Time Current"
            
        elif format == 'product':
            operatorName = r"$\int dt\, \langle j(t) \rangle \langle j(t + \tau) \rangle$"
            data = currentData.doubleTimeCurrentProduct
            fileName = "Double-Time Current Product"

        elif format == 'connected':
            operatorName = r"$\int dt\, \langle j(t) j(t + \tau) \rangle - \langle j(t) \rangle \langle j(t + \tau) \rangle$"
            data = currentData.timeConnectedCorrelator
            fileName = "Connected Current Correlator"
    
        else:
            raise ValueError(f"The value of format must be 'noise', 'product', or 'connected', not '{format}'.")
        
        self.__PlotOneComplexFunction(
            x = self.__axes.tauAxisDim,
            y = data,
            dataName = operatorName,
            title = self.__GenerateTitle(self.__sim.momentums)
        )

        if saveFigs:
            plt.savefig(f"{self.__plotFolder}/{fileName}.png", dpi=300)
        if show:
            plt.show() 

    def PlotCurrentFFT(self, saveFigs: bool=False, show: bool=True, overplotHarmonicLines: bool=True, fLim: tuple[float, float]=(-12.5, 12.5)) -> None:
        """
        Plots the FFT of the current operator.

        Parameters
        ----------
        saveFigs : bool
            Whether to save the figures.
        show : bool
            Whether to show the figures.
        overplotHarmonicLines : bool
            Whether to overplot vertical lines at the harmonic frequencies.
        fLim : tuple[float, float]
            A tuple containing the minimum and maximum frequency limits to plot.
        """

        currentData = self.__sim.totalCurrent
        plt.semilogy(self.__axes.freqAxis,
                     currentData.freqDomainCurrent,
                     color = 'black')
    
        plt.xlabel(self.__fLabel)
        plt.ylabel(fr"FFT of $\langle j(t) \rangle$")
        plt.title(self.__GenerateTitle(self.__sim.momentums))
        
        if overplotHarmonicLines:
            # Adds dashed lines at the harmonics.       
            for n in range(np.ceil((fLim[0],)), np.floor((fLim[1],))):
                plt.axvline(n, color='blue', linestyle='dashed')

        if saveFigs:
            plt.savefigs(f"{self.__plotFolder}/Current FFT.png", dpi=300)
        if show:
            plt.show()

    def PlotConnectedCurrentFFT(self, saveFigs: bool=False, show: bool=True, overplotHarmonicLines: bool=True, fLim: tuple[float, float]=(-12.5, 12.5)) -> None:
        """
        Plots the FFT of the connected current correlator.

        Parameters
        ----------
        saveFigs : bool
            Whether to save the figures.
        show : bool
            Whether to show the figures.
        overplotHarmonicLines : bool
            Whether to overplot vertical lines at the harmonic frequencies.
        fLim : tuple[float, float]
            A tuple containing the minimum and maximum frequency limits to plot.
        """

        currentData = self.__sim.totalCurrent
        plt.semilogy(self.__axes.freqAxis,
                     currentData.freqConnectedCorrelator.real,
                     color = 'black')
    
        plt.xlabel(self.__fLabel)
        plt.ylabel(fr"Real Part of FFT of $\int dt\, \langle j(t) j(t + \tau) \rangle - \langle j(t) \rangle \langle j(t + \tau) \rangle$")
        plt.title(self.__GenerateTitle(self.__sim.momentums))
        
        if overplotHarmonicLines:
            # Adds dashed lines at the harmonics.       
            for n in range(np.ceil((fLim[0],)), np.floor((fLim[1],))):
                plt.axvline(n, color='blue', linestyle='dashed')

        if saveFigs:
            plt.savefigs(f"{self.__plotFolder}/Current Connected Correlator FFT.png", dpi=300)
        if show:
            plt.show()

    def PlotHarmonics(self, saveFigs: bool=False, show: bool=True) -> None:
        """
        Plots the magnitude, real, and imaginary part of the numerically integrated Fourier transform at the harmonics
        of the driving frequency on separate plots.

        Parameters
        ----------
        saveFigs : bool
            Whether to save the figures.
        show : bool
            Whether to show the figures.
        """

        currentData = self.__sim.totalCurrent
        labels = ['Magnitude', 'Real', 'Imag']
        
        # Loops through all the plotting functions.
        for funcIndex in range(len(self.__plottingFunctions)):
            # Figures out how many harmonics we have.
            n = (currentData.harmonics.size - 1) // 2

            # Isolates the positive values.
            data = self.__plottingFunctions[funcIndex](currentData.harmonics)
            positiveMask = data >= 0

            # Plots positive values in black.
            plt.semilogy(np.arange(-n, n + 1),
                        data[positiveMask],
                        'o',
                        color='black')

            # Plots negative values in red.
            plt.semilogy(np.arange(-n, n + 1),
                         data[~positiveMask],
                         'o',
                         color='red')
            
            plt.xlabel(self.__fLabel)
            plt.ylabel(rf"{self.__plottingPrefixes} FFT of $\int dt\, \langle j(t) j(t + \tau) \rangle - \langle j(t) \rangle \langle j(t + \tau) \rangle$")

            if saveFigs:
                plt.savefig(f"{self.__plotFolder}/[{labels[funcIndex]}] Current Connected Correlator FFT.png", dpi=300)
            if show:
                plt.show()

    def __PlotOneComplexFunction(self, x: np.ndarray[float], y: np.ndarray[complex], dataName: str | None, title: str | None, createFigure: bool=False) -> None:
        """
        Plots a single, one-dimensional complex function. Forms a 3 row by 1 column subplot,
        where the rows correspond to the magnitude, real part, and imaginary part of the function.

        Doesn't save or show the figure - that is the job of the parent function.
        
        Parameters
        ----------
        x : ndarray[float]
            The x-axis of the data.
        y : ndarray[complex]
            The complex data to plot.
        dataName : str | None
            The name or label to give the data on the y-axis. Not used if createFigure is false.
        title : str | None
            The title of the plot. Not used if createFigure is False.
        createFigure : bool
            Whether to create a new figure. If false, will just plot onto the current plot.
        """

        nrows, ncols = 3, 1
        if createFigure:
            fig, ax = plt.subplots(nrows, ncols)
            plt.suptitle(title)

        for row in range(nrows):
            ax[row].plot(x,
                         self.__plottingFunctions[row](y))
            
            if createFigure:
                ax[row].set_xlabel(self.__tLabel)
                ax[row].set_ylabel(rf"{self.__plottingPrefixes[row]} {dataName}")