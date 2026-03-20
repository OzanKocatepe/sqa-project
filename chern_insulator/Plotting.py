import numpy as np
import matplotlib.pyplot as plt
import os

from Ensemble import Ensemble

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
        plt.style.use('stylesheet.mplstyle')

        # Defines useful attributes for plotting.
        self.__plottingFunctions = [lambda z: np.abs(z), lambda z: z.real, lambda z: z.imag]
        self.__plottingLabels = ["Magnitude", "Real", "Imaginary"]
        self.__tLabel = r"$t \gamma_-$"
        self.__freqLabel = r"$f / \Omega$"
        self.__plotFolder = "chern_insulator/plots"

    def PlotParamagneticCurrent(self) -> None:
        """Plots the paramagnetic current as a function of time."""

        axes = self.__ensemble.axes
        current = self.__ensemble.totalCurrent

        labels = [r"$\hat j_x$", r"$\hat j_y$"]
        alpha = 0.2
        colors = ['tab:blue', 'orange']

        for operatorIndex in np.arange(2).astype(int):
            # Stores the index of the operator that's going to be 
            # plotted in low opacity.
            opacityIndex = 1 if operatorIndex == 0 else 0
            fig, ax = plt.subplots(2, 1)

            # Plots the main operator with full opacity.
            ax[0].plot(axes.tauAxisDim,
                       current.paramagneticCurrent[operatorIndex].real,
                       color = colors[operatorIndex],
                       label = labels[operatorIndex])

            ax[1].plot(axes.tauAxisDim,
                       current.paramagneticCurrent[operatorIndex].imag,
                       color = colors[operatorIndex],
                       label = labels[operatorIndex])

            # Plots the other operator with small opacity.
            ax[0].plot(axes.tauAxisDim,
                       current.paramagneticCurrent[opacityIndex].real,
                       color = colors[opacityIndex],
                       label = labels[opacityIndex],
                       alpha = alpha)

            ax[1].plot(axes.tauAxisDim,
                       current.paramagneticCurrent[opacityIndex].imag,
                       color = colors[opacityIndex],
                       label = labels[opacityIndex],
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
            folder = f"{self.__plotFolder}/Delta {self.__ensemble.params.delta}/Paramagnetic Current"
            os.makedirs(folder, exist_ok=True)
            # Saves.
            plt.savefig(f"{folder}/J{['x', 'y'][operatorIndex]}", dpi=300)

            plt.show()
            plt.close()


    def PlotParamagneticCurrentFFT(self) -> None:
        """Plots the FFT of the paramagnetic current as a function of frequency."""

        axes = self.__ensemble.axes
        current = self.__ensemble.totalCurrent

        jxFFT = np.fft.fftshift(np.fft.fft(current.paramagneticCurrent[0]))
        jyFFT = np.fft.fftshift(np.fft.fft(current.paramagneticCurrent[1]))

        plt.semilogy(
            axes.freqAxis,
            np.abs(jxFFT),
            label=r'$\hat j_x$',
            color='tab:blue')

        plt.semilogy(
            axes.freqAxis,
            np.abs(jyFFT),
            label=r'$\hat j_y$',
            color='orange')

        plt.xlim(-11, 11)
        plt.xlabel(r"$f / \Omega$")
        plt.ylabel("Magnitude of FFT")
        plt.legend()

        for n in np.arange(-10, 11):
            plt.axvline(n, color='black', linestyle='dashed', alpha=0.2)

        plt.title(fr"Current Operators with $\Delta = {self.__ensemble.params.delta}$")

        plt.tight_layout()

        # Makes subfolder if it doesn't exist.
        folder = f"{self.__plotFolder}/Delta {self.__ensemble.params.delta}/Paramagnetic Current"
        os.makedirs(folder, exist_ok=True)
        # Saves.
        plt.savefig(f"{folder}/Current FFT", dpi=300)

        plt.show()
        plt.close()