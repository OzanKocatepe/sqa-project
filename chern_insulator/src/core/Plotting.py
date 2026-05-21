import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import integrate

from .Ensemble import Ensemble
from config import PLOTTING_DIR, STYLESHEET

class Plotting:
    """Contains the logic for plotting the results of our simulations."""

    def __init__(self, ensemble: Ensemble) -> None:
        """
        Initialises the instance.

        Parameters
        ----------
        ensemble : Ensemble
            The ensemble that we are getting the plotting data from.
        """

        self.__ensemble = ensemble

        # Sets the style sheet.
        plt.style.use(STYLESHEET)

        # Defines useful attributes for plotting.
        self.__plottingFunctions = [lambda z: np.abs(z), lambda z: z.real, lambda z: z.imag]
        self.__plottingLabels = ["Magnitude", "Real", "Imaginary"]
        self.__tauLabel = r"$\tau \gamma_-$"
        self.__tLabel = r"$t \gamma_-$"
        self.__freqLabel = r"$f / \Omega$"

    def PlotSingleTime(self, kx: float, ky: float, tMax: float=None, overplotNumericalSolution: bool=False) -> None:
        """
        Plots the single-time correlations as functions of time.
        
        Parameters
        ----------
        kx : float
            The x-component of the momentum to plot.
        ky : float
            The y-component of the momentum to plot.
        tMax : float
            The non-dimensional time to plot until.
        overplotNumericalSolution : bool
            Whether to overplot the numerical simulation, solved using the equations
            of motion ODE and numerical approximation methods (solve_ivp from scipy)
            to confirm that the Fourier series are accurate.
        """

        axes = self.__ensemble.axes
        model = self.__ensemble.models[(kx, ky)]
        sigma = model.correlationData.singleTimeFourier
        functions = self.__plottingFunctions[1:]
        functionLabels = ['Real', 'Imaginary']
        
        subscripts = ['-', '+', 'z']

        fig, ax = plt.subplots(2, 3)

        for row in range(2):
            for col in range(3):
                ax[row, col].plot(axes.tauAxisDim,
                                  functions[row](sigma[col].Evaluate(axes.tauAxisSec)),
                                  color = 'black', label='Fourier Series')
                
                if overplotNumericalSolution:
                    from operators import Hamiltonian
                    h = Hamiltonian(model.params)

                    numericalSigma = integrate.solve_ivp(
                        fun = h.EquationsOfMotion,
                        t_span = (0, np.max(axes.tauAxisSec)),
                        y0 = np.array([0.0, 0.0, -1.0], dtype=complex),
                        t_eval = axes.tauAxisSec,
                        rtol=1e-9,
                        atol=1e-12,
                        vectorized = True
                    ).y

                    ax[row, col].plot(axes.tauAxisDim,
                                      functions[row](numericalSigma[col]),
                                      color = 'blue', label='Numerical Soln')
                    ax[row, col].legend()

                ax[row, col].set_xlabel(self.__tLabel)
                ax[row, col].set_ylabel(fr"{functionLabels[row]} Part of $\hat \sigma_{subscripts[col]}$")
                ax[row, col].set_xlim(None, tMax)
        
        plt.suptitle(rf"Single-Time Correlations for $(k_x, k_y) = ({kx / np.pi:.2f}\pi, {ky / np.pi:.2f}\pi)$")
        plt.tight_layout()
        
        folder = f"{PLOTTING_DIR}/Delta {self.__ensemble.params.delta}/Single-Time"
        os.makedirs(folder, exist_ok=True)
        plt.savefig(f"{folder}/kx: {kx / np.pi}pi, ky: {ky / np.pi}pi.png", dpi=300)
        plt.show()
        plt.close()

    def PlotDoubleTimeCorrelation(self, kx: float, ky: float) -> None:
        """Plots the double-time correlations.
        
        For each double-time correlation sigma_i sigma_j, we calculate
        and plot the connected correlators. Each connected correlators
        has a real and imaginary part plotted side-by-side on a 2D plot.

        Parameters
        ----------
        kx : float
            The x-component of the momentum to plot.
        ky : float
            The y-component of the momentum to plot.
        """

        corr = self.__ensemble.models[(kx, ky)].correlationData
        axes = self.__ensemble.axes
        subscripts = ['-', '+', 'z']

        for leftOperatorIndex in range(3):
            for rightOperatorIndex in range(3):
                leftOperator = corr.singleTimeFourier[leftOperatorIndex].Evaluate(axes.tAxisSec)
                # Evaluates the right operator at every point (t + tau).
                rightOperator = corr.singleTimeFourier[rightOperatorIndex].Evaluate(
                    # Creates an object of (t.size, tau.size) containing pairs (t, t + tau).
                    np.add.outer(axes.tAxisSec, axes.tauAxisSec)
                )

                # Left operator is evaluated at t, so stays the same as right operator changes with tau.
                # This calcalates <sigma_i> <sigma_j> at times (t, t + tau).
                prod = leftOperator[:, np.newaxis] * rightOperator
                connectedCorr = corr.doubleTimeCorrelations[leftOperatorIndex, rightOperatorIndex, :, :] - prod

                fig, ax = plt.subplots(1, 2)
                funcs = self.__plottingFunctions[1:]
                
                for col in range(2):
                    mesh = ax[col].pcolormesh(
                        self.__ensemble.axes.tauAxisDim,
                        self.__ensemble.axes.tAxisDim,
                        funcs[col](connectedCorr),
                        cmap = 'bwr',
                        shading = 'nearest')

                    ax[col].set_title(f"{self.__plottingLabels[col + 1]} Part")
                    ax[col].set_xlabel(self.__tauLabel)
                    ax[col].set_ylabel(self.__tLabel)
                    plt.colorbar(mesh, ax=ax[col])

                plt.suptitle(f"{subscripts[leftOperatorIndex]}{subscripts[rightOperatorIndex]} "
                            + fr"Connected Correlator ($\Delta = {self.__ensemble.params.delta}$)")
                plt.tight_layout()

                subFolder = f"Delta {self.__ensemble.params.delta}/Double-Time Connected Correlators"
                os.makedirs(f"{PLOTTING_DIR}/{subFolder}", exist_ok=True)
                plt.savefig(f"{PLOTTING_DIR}/{subFolder}/"
                            + f"{subscripts[leftOperatorIndex]}{subscripts[rightOperatorIndex]}.png", dpi=300)
                plt.show()

    def PlotDoubleTimeCurrent(self) -> None:
        """Plots the double-time current.
        
        For each double-time current direction (xx, xy, yx, xx),
        we plot the real and imaginary components on their own axes.
        """

        curr = self.__ensemble.meanCurrent
        axes = self.__ensemble.axes
        subscripts = ['x', 'y']

        for leftOperatorIndex in range(2):
            for rightOperatorIndex in range(2):
                fig, ax = plt.subplots(1, 2)
                funcs = self.__plottingFunctions[1:]
                
                for col in range(2):
                    mesh = ax[col].pcolormesh(
                        self.__ensemble.axes.tauAxisDim,
                        self.__ensemble.axes.tAxisDim,
                        funcs[col](curr.doubleTimeCurrent[leftOperatorIndex, rightOperatorIndex]),
                        cmap = 'bwr',
                        shading = 'nearest')

                    ax[col].set_title(f"{self.__plottingLabels[col + 1]} Part")
                    ax[col].set_xlabel(self.__tauLabel)
                    ax[col].set_ylabel(self.__tLabel)
                    plt.colorbar(mesh, ax=ax[col])

                plt.suptitle(f"{subscripts[leftOperatorIndex]}{subscripts[rightOperatorIndex]} "
                            + fr"Connected Correlator ($\Delta = {self.__ensemble.params.delta}$)")
                plt.tight_layout()

                subFolder = f"Delta {self.__ensemble.params.delta}/Double-Time Connected Current"
                os.makedirs(f"{PLOTTING_DIR}/{subFolder}", exist_ok=True)
                plt.savefig(f"{PLOTTING_DIR}/{subFolder}/"
                            + f"{subscripts[leftOperatorIndex]}{subscripts[rightOperatorIndex]}.png", dpi=300)
                plt.show()
        

    def PlotTotalCurrent(self, overplotLengthGauge: bool=False) -> None:
        """Plots the total current as a function of time."""

        axes = self.__ensemble.axes
        current = self.__ensemble.meanCurrent.totalCurrent
        lengthCurrent = self.__ensemble.meanCurrent.lengthGaugeCurrent

        labels = [r"$\hat j_x$", r"$\hat j_y$"]
        alpha = 0.2
        colors = ['tab:blue', 'orange']
        lengthGaugeColors = ['purple', 'red']

        for operatorIndex in np.arange(2).astype(int):
            # Stores the index of the operator that's going to be 
            # plotted in low opacity.
            opacityIndex = 1 if operatorIndex == 0 else 0
            fig, ax = plt.subplots(2, 1)

            # Plots the main operator with full opacity.
            ax[0].plot(axes.tauAxisDim,
                       current[operatorIndex].real,
                       color = colors[operatorIndex],
                       label = labels[operatorIndex])

            ax[1].plot(axes.tauAxisDim,
                       current[operatorIndex].imag,
                       color = colors[operatorIndex],
                       label = labels[operatorIndex]) 

            # Plots the other operator with small opacity.
            ax[0].plot(axes.tauAxisDim,
                       current[opacityIndex].real,
                       color = lengthGaugeColors[opacityIndex],
                       label = labels[opacityIndex],
                       alpha = alpha)

            ax[1].plot(axes.tauAxisDim,
                       current[opacityIndex].imag,
                       color = lengthGaugeColors[opacityIndex],
                       label = labels[opacityIndex],
                       alpha = alpha)
            
            if overplotLengthGauge:
                ax[0].plot(axes.tauAxisDim,
                        lengthCurrent[operatorIndex].real,
                        color = lengthGaugeColors[operatorIndex],
                        label = f"{labels[operatorIndex]} (LG)")

                ax[1].plot(axes.tauAxisDim,
                        lengthCurrent[operatorIndex].imag,
                        color = lengthGaugeColors[operatorIndex],
                        label = f"{labels[operatorIndex]} (LG)")

                # Plots the other operator with small opacity.
                ax[0].plot(axes.tauAxisDim,
                        lengthCurrent[opacityIndex].real,
                        color = colors[opacityIndex],
                        label = f"{labels[opacityIndex]} (LG)",
                        alpha = alpha)

                ax[1].plot(axes.tauAxisDim,
                        lengthCurrent[opacityIndex].imag,
                        color = colors[opacityIndex],
                        label = f"{labels[opacityIndex]} (LG)",
                        alpha = alpha)
            
            # Sets x-axes to dimensionless time.
            ax[0].set_xlabel(self.__tLabel)
            ax[1].set_xlabel(self.__tLabel)

            # Sets y-axes to real and imaginary components.
            ax[0].set_ylabel("Real Part of Operator")
            ax[1].set_ylabel("Imaginary Part of Operator")

            # Turns on the legend.
            ax[0].legend()
            ax[1].legend()
            
            plt.suptitle(rf"Current Operator {labels[operatorIndex]} with $\Delta = {self.__ensemble.params.delta}$")

            # Makes subfolder if it doesn't exist.
            folder = f"{PLOTTING_DIR}/Delta {self.__ensemble.params.delta}/Paramagnetic Current"
            os.makedirs(folder, exist_ok=True)
            # Saves.
            plt.savefig(f"{folder}/J{['x', 'y'][operatorIndex]}", dpi=300)

            plt.show()
            plt.close()

    def PlotIntegratedSecondOrderCurrent(self) -> None:
        """Plots the integrated second order currents on a single plot, but saves/shows
        it as four separate plots with different currents highlighted.
        """

        alpha = 0.2
        labels = [r"$\langle D_{x, x} \rangle_t$", r"$\langle D_{x, y} \rangle_t$", r"$\langle D_{y, x} \rangle_t$", r"$\langle D_{y, y} \rangle_t$"]
        fileNames = ["xx", "xy", "yx", "yy"]
        colors = ['tab:blue', 'orange', 'purple', 'red']
        indices = [(0, 0), (0, 1), (1, 0), (1, 1)]

        axes = self.__ensemble.axes
        current = self.__ensemble.meanCurrent.meanSecondOrderCurrent
        
        for highlightedIndex in range(4):
            fig, ax = plt.subplots(2, 1)

            # Plots the highlighted operator with full transparency.
            ax[0].plot(
                axes.tauAxisDim,
                current[indices[highlightedIndex]].real,
                color = colors[highlightedIndex],
                label = labels[highlightedIndex]
            )

            ax[1].plot(
                axes.tauAxisDim,
                current[indices[highlightedIndex]].imag,
                color = colors[highlightedIndex],
                label = labels[highlightedIndex]
            )

            for otherIndex in range(4):
                # Skips the highlighted index.
                if otherIndex == highlightedIndex:
                    continue

                ax[0].plot(
                    axes.tauAxisDim,
                    current[indices[otherIndex]].real,
                    color = colors[otherIndex],
                    label = labels[otherIndex],
                    alpha = alpha
                )

                ax[1].plot(
                    axes.tauAxisDim,
                    current[indices[otherIndex]].imag,
                    color = colors[otherIndex],
                    label = labels[otherIndex],
                    alpha = alpha
                )

            ax[0].legend()
            ax[0].set_xlabel(self.__tauLabel)
            ax[0].set_ylabel("Real Part")

            ax[1].legend()
            ax[1].set_xlabel(self.__tauLabel)
            ax[1].set_ylabel("Imaginary Part")

            plt.tight_layout()
            folder = PLOTTING_DIR / f"Delta {self.__ensemble.params.delta}/Int. 2nd Order Current"
            os.makedirs(folder, exist_ok=True)
            plt.savefig(f"{folder}/{fileNames[highlightedIndex]}.png", dpi=300)
            plt.show() 

    def PlotTotalCurrentFFT(self, linearScale: bool=False, overplotLengthGauge: bool=False) -> None:
        """Plots the FFT of the total current as a function of frequency.
        
        Parameters
        ----------
        linearScale : bool
            A debug option to scale the axes linearly, giving a more obvious view of
            the relative scales of the Jx and Jy peaks. False by default.
        """

        axes = self.__ensemble.axes
        current = self.__ensemble.meanCurrent

        jxFFT = np.fft.fftshift(np.fft.fft(current.totalCurrent[0]))
        jyFFT = np.fft.fftshift(np.fft.fft(current.totalCurrent[1]))

        plt.plot(
            axes.freqAxis,
            np.abs(jxFFT),
            label=r'$\hat j_x$',
            color='tab:blue')

        plt.plot(
            axes.freqAxis,
            np.abs(jyFFT),
            label=r'$\hat j_y$',
            color='orange')
        
        if overplotLengthGauge:
            jxLGFFT = np.fft.fftshift(np.fft.fft(current.lengthGaugeCurrent[0]))
            jyLGFFT = np.fft.fftshift(np.fft.fft(current.lengthGaugeCurrent[1]))

            plt.plot(
                axes.freqAxis,
                np.abs(jxLGFFT),
                label=r'$\hat j_x$ (LG)',
                color='purple')

            plt.plot(
                axes.freqAxis,
                np.abs(jyLGFFT),
                label=r'$\hat j_y$ (LG)',
                color='red')
        
        if not linearScale:
            plt.yscale('log')

        plt.xlim(-11, 11)
        plt.xlabel(r"$f / \Omega$")
        plt.ylabel("Magnitude of FFT")
        plt.legend()

        for n in np.arange(-10, 11):
            plt.axvline(n, color='black', linestyle='dashed', alpha=0.2)

        plt.title(fr"Current Operators with $\Delta = {self.__ensemble.params.delta}$")
        plt.tight_layout()

        # Makes subfolder if it doesn't exist.
        folder = f"{PLOTTING_DIR}/Delta {self.__ensemble.params.delta}/Paramagnetic Current"
        os.makedirs(folder, exist_ok=True)
        # Saves.
        plt.savefig(f"{folder}/Current FFT", dpi=300)

        plt.show()
        plt.close()