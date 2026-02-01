import numpy as np
import matplotlib.pyplot as plt
from .SSHSimulation import SSHSimulation

class SSHVisualiser:
    """Manages the plotting logic for the information stored in SSHSimulation."""

    def __init__(self, sim: SSHSimulation):
        """Constructs an instance of the visualiser.
        
        Parameters
        ----------
        sim : SSHSimulation
            The instance of SSHSimulation which stores all of the information that we
            may want to plot. Should have already had Run() called, since that relies on
            other user inputs.
        """

        self._sim = sim
        self._plottingFunctions = [lambda z: np.abs(z), lambda z: z.real, lambda z: z.imag]
        self._tLabel = r"$t \gamma_-$"
        self._tauLabel = r"$\tau \gamma_-$"

    def PlotSingleTimeCorrelations(self, k: float, overplotFourier: bool=False, saveFigs: bool=False) -> None:
        r"""Plots the single-time correlations $\langle \tilde \sigma_-(t) \rangle,\, \langle \tilde \sigma_+(t) \rangle,\, \langle \tilde \sigma_z(t) \rangle$ for a fixed momentum.

        Parameters
        ----------
        k : float
            The momentum for which we want to plot the correlation functions.
        overplotFourier : bool
            Whether to overplot the fourier expansions as well.
        saveFigs : bool
            Determines whether to save the figure or not.
        """

        correlationData = self._sim.models[k].correlationData

        expectationLabels = [r"$\langle \tilde \sigma_-(t) \rangle$",
                             r"$\langle \tilde \sigma_+(t) \rangle$",
                             r"$\langle \tilde \sigma_z(t) \rangle$"]
                
        yLabels = []
        for i in range(len(expectationLabels)):
            yLabels.append(
                [f"Magnitude of {expectationLabels[i]}",
                f"Real Part of {expectationLabels[i]}",
                f"Imaginary Part of {expectationLabels[i]}"]
            )

        # Creates the 3x3 subplots.
        nrows, ncols = 3, 3
        fig, ax = plt.subplots(nrows, ncols, figsize=(16, 8.8))

        for row in np.arange(nrows):
            for col in np.arange(ncols):
                # Plot numerical solution.
                ax[row, col].plot(correlationData.tauAxisDim, self._plottingFunctions[col](correlationData.singleTime[row]),
                                color = "Black")

                if overplotFourier:
                    ax[row, col].plot(correlationData.tauAxisDim, self._plottingFunctions[col](correlationData.singleTimeFourier[row].Evaluate(correlationData.tauAxisSec)),
                                    color = "blue")
        
                # Sets other properties.
                ax[row, col].set_xlabel(self._tLabel)
                ax[row, col].set_ylabel(yLabels[row][col])

        title = rf"$k = {k / np.pi} \pi,\, t_1 = {self._sim.params.t1},\, t_2 = {self._sim.params.t2},\, A_0 = {self._sim.params.drivingAmplitude},\, \Omega = {self._sim.params.drivingFreq:.5f},\, \gamma_- = {self._sim.params.decayConstant}$"
        plt.suptitle(title)
        plt.tight_layout()
        if saveFigs:
            plt.savefig("plots/Single-Time Correlators.png", dpi=300)
        plt.show()

    def PlotDoubleTimeCorrelations(self, k: float, slice: list[tuple[int]]=None, numTauPoints: int=None, saveFigs: bool=False, subtractUncorrelatedValues: bool=False, vLim: tuple[float]=(None, None)) -> None:
        r"""Plots the double-time correlations.

        Parameters
        ----------
        k : float
            The momentum for which we want to plot the correlation functions.
        slice : list[tuple[int]]
            A list of tuples of the form (i, j), which will make the function only
            plot those specific double-time correlations.
        numTauPoints : int
            The number of tau points to plot on the 3D figures. If none, will just use the normal tauAxis.
        saveFigs : bool
            Determines whether to save the figure or not.
        subtractUncorrelatedValues : bool
            Whether, for the double-time correlation $\langle \sigma_i(t), \sigma_j(t + \tau) \rangle$, to subtract the value of
            $\langle \sigma_i (t) \rangle \langle \sigma_j(t + \tau) \rangle$. This would be the value of the double-time correlation if the two operators were
            entirely uncorrelated, and as $\tau$ gets sufficiently large we expect the system to become uncorrelated due to
            interaction with the environment.
        vLim : tuple[float]
            The limits of the colorbar to use when plotting.
        """

        correlationData = self._sim.models[k].correlationData

        tauMask = None
        if numTauPoints is None:
            tauMask = np.ones((correlationData.tauAxisDim.size,), dtype=bool)
        else:
            modulus = self._sim.tauAxisDim.size // numTauPoints
            tauMask = np.arange(correlationData.tauAxisDim.size) % modulus == 0

        subscripts = ['-', '+', 'z']

        operatorLabels = [r"\tilde \sigma_-",
                          r"\tilde \sigma_+",
                          r"\tilde \sigma_z"]

        # Loops over all nine double-time correlation functions.
        if slice is None:
            i, j = np.meshgrid(range(3), range(3))
            iterable = tuple(zip(i.flatten(), j.flatten()))
        # Or, allow us to choose a subset to plot.
        else:
            iterable = slice

        for i, j in iterable:
                # Creates the y-labels for each pair of operators.
                correlationName = rf"$\langle {operatorLabels[i]}(t) {operatorLabels[j]}(t + \tau) \rangle$" 
                if subtractUncorrelatedValues:
                    correlationName = rf"$\langle {operatorLabels[i]}(t) {operatorLabels[j]}(t + \tau) \rangle - \langle {operatorLabels[i]}(t) \rangle \langle {operatorLabels[j]}(t + \tau) \rangle$"

                zLabels = [
                    f"Real Part of {correlationName}",
                    f"Imaginary Part of {correlationName}"
                ] 

                # Creates the 2 3D subplots.
                nrows, ncols = 1, 2
                fig, ax = plt.subplots(nrows, ncols, figsize=(16, 8.8))

                for col in np.arange(ncols):                    
                    # Plots each system as a line, with each line representing
                    # a different initial condition within a steady-state period.
                    z = np.zeros((correlationData.tAxisSec.size, correlationData.tauAxisSec.size), dtype=complex)
                    for tIndex, t in enumerate(correlationData.tAxisSec):
                        z[tIndex, :] = correlationData.doubleTime[i, j, tIndex, :]
                        
                        # Subtracts the uncorrelated values if the system desired that.
                        if subtractUncorrelatedValues:
                            # Subtracts the value of the single-time correlation fourier expansions
                            # at each time on the tau axis.
                            newAxis = t + correlationData.tauAxisSec
                            z[tIndex, :] -= correlationData.singleTimeFourier[i].Evaluate(t)[0] * correlationData.singleTimeFourier[j].Evaluate(newAxis)

                    mesh = ax[col].pcolormesh(correlationData.tauAxisDim[tauMask], correlationData.tAxisDim, self._plottingFunctions[1:][col](z)[:, tauMask], shading='nearest', cmap='bwr', vmin=vLim[0], vmax=vLim[1])
        
                    # Sets other properties.
                    ax[col].set_xlabel(self._tauLabel)
                    ax[col].set_ylabel(self._tLabel) 
                    fig.colorbar(mesh, ax=ax[col])

                title = rf"{correlationName} -- $k = {k / np.pi} \pi,\, t_1 = {self._sim.params.t1},\, t_2 = {self._sim.params.t2},\, A_0 = {self._sim.params.drivingAmplitude},\, \Omega = {self._sim.params.drivingFreq:.5f},\, \gamma_- = {self._sim.params.decayConstant}$"
                plt.suptitle(title)
                plt.tight_layout()

                if saveFigs:
                    folderName = "Double-Time Correlators"
                    if subtractUncorrelatedValues:
                        folderName = "Double-Time Connected Correlators"
                    plt.savefig(f"plots/{folderName}/ssh {subscripts[i]}, {subscripts[j]}.png", dpi=300)

                plt.show()

    def PlotSingleTimeProducts(self, k: float, slice: list[tuple[int]]=None, numTauPoints: int=None, saveFigs: bool=False, vLim: tuple[float]=(None, None)) -> None:
        r"""
        Plots the expected steady state of the double-time correlations, which is the product of the
        expectations of the single-time correlations at times $t$ and $t' = t + \tau$.

        Parameters
        ----------
        k : float
            The momentum for which we want to plot the correlation functions.
        slice : list[tuple[int]]
            A list of tuples of the form (i, j), which will make the function only
            plot those specific double-time correlations.
        numTauPoints : int
            The number of tau points to plot on the 3D figures. If none, will just use the normal tauAxis.
        saveFigs : bool
            Determines whether to save the figure or not.
        vLim : tuple[float]
            The limits of the colorbar to use when plotting.
        """

        correlationData = self._sim.models[k].correlationData

        tauMask = None
        if numTauPoints is None:
            tauMask = np.ones((correlationData.tauAxisDim.size,), dtype=bool)
        else:
            modulus = self._sim.tauAxisDim.size // numTauPoints
            tauMask = np.arange(correlationData.tauAxisDim.size) % modulus == 0

        subscripts = ['-', '+', 'z']

        operatorLabels = [r"\tilde \sigma_-",
                          r"\tilde \sigma_+",
                          r"\tilde \sigma_z"]

        # Loops over all nine double-time correlation functions.
        if slice is None:
            i, j = np.meshgrid(range(3), range(3))
            iterable = tuple(zip(i.flatten(), j.flatten()))
        # Or, allow us to choose a subset to plot.
        else:
            iterable = slice

        for i, j in iterable:
                # Creates the y-labels for each pair of operators.
                correlationName = rf"$\langle {operatorLabels[i]}(t) \rangle \langle {operatorLabels[j]}(t + \tau) \rangle$"
                zLabels = [
                    f"Real Part of {correlationName}",
                    f"Imaginary Part of {correlationName}"
                ] 

                # Creates the 2 3D subplots.
                nrows, ncols = 1, 2
                fig, ax = plt.subplots(nrows, ncols, figsize=(16, 8.8))

                for col in np.arange(ncols):                    
                    # Plots each system as a line, with each line representing
                    # a different initial condition within a steady-state period.
                    z = np.zeros((correlationData.tAxisSec.size, correlationData.tauAxisSec.size), dtype=complex)
                    for tIndex, t in enumerate(correlationData.tAxisSec):
                        newAxis = t + correlationData.tauAxisSec
                        z[tIndex, :] = correlationData.singleTimeFourier[i].Evaluate(t)[0] * correlationData.singleTimeFourier[j].Evaluate(newAxis)

                    mesh = ax[col].pcolormesh(correlationData.tauAxisDim[tauMask], correlationData.tAxisDim, self._plottingFunctions[1:][col](z)[:, tauMask], shading='nearest', cmap='bwr', vmin=vLim[0], vmax=vLim[1])
        
                    # Sets other properties.
                    ax[col].set_xlabel(self._tLabel)
                    ax[col].set_ylabel(self._tauLabel)
                    plt.colorbar(mesh, ax=ax[col])

                title = rf"{correlationName} -- $k = {k / np.pi} \pi,\, t_1 = {self._sim.params.t1},\, t_2 = {self._sim.params.t2},\, A_0 = {self._sim.params.drivingAmplitude},\, \Omega = {self._sim.params.drivingFreq:.5f},\, \gamma_- = {self._sim.params.decayConstant}$"
                plt.suptitle(title)
                plt.tight_layout()
                if saveFigs:
                    plt.savefig(f"plots/Single-Time Products/ssh {subscripts[i]} * {subscripts[j]}.png", dpi=300)
                plt.show()
        
    def PlotTotalCurrent(self) -> None:
        """Plots the total current operator in the time and frequency domains."""

        currentData = self._sim.CalculateTotalCurrent()

        kValues = self._sim.momentums
        title = rf"$k = {kValues} \pi,\, t_1 = {self._sim.params.t1},\, t_2 = {self._sim.params.t2},\, A_0 = {self._sim.params.drivingAmplitude},\, \Omega = {self._sim.params.drivingFreq:.5f},\, \gamma_- = {self._sim.params.decayConstant}$"

        currentLabel = r"$\langle\tilde j_k \rangle$"
        yLabels = [
            f"Magnitude of {currentLabel}",
            f"Real Part of {currentLabel}",
            f"Imaginary Part of {currentLabel}"
        ]

        nrows, ncols = 3, 1
        fig, ax = plt.subplots(nrows, ncols, figsize=(16, 8.8))

        for row in np.arange(nrows):
            # Plot the numerical solution.
            ax[row].plot(currentData.tauAxisDim, self._plottingFunctions[row](currentData.timeDomainData),
                        color = "Black")
    
            ax[row].set_xlabel(self._tLabel)
            ax[row].set_ylabel(yLabels[row])

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(16, 8.8))

        # Plotting the fourier transform of the current operator.
        plt.semilogy(currentData.freqAxis / self._sim.params.drivingFreq, np.abs(currentData.freqDomainData)**2,
                color = 'black')

        plt.suptitle(title)
        plt.xlabel(r"$\omega / \Omega$")
        plt.ylabel(r"$\| \tilde j (\omega) \|^2$")
        plt.show()