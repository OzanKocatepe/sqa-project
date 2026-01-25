import numpy as np
import matplotlib.pyplot as plt
from SSHSimulation import SSHSimulation

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

    def PlotSingleTimeCorrelations(self, k: float):
        r"""Plots the single-time correlations $\langle \tilde \sigma_-(t) \rangle,\, \langle \tilde \sigma_+(t) \rangle,\, \langle \tilde \sigma_z(t) \rangle$ for a fixed momentum.

        Parameters
        ----------
        k : float
            The momentum for which we want to plot the correlation functions.
        """

        model = self._sim._models[k]

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
                ax[row, col].plot(self._sim.tAxis, self._plottingFunctions[col](model.solution.y[row]),
                                color = "Black")
        
                # Sets other properties.
                ax[row, col].set_xlabel(self._tLabel)
                ax[row, col].set_ylabel(yLabels[row][col])

        title = rf"$k = {k / np.pi} \pi,\, t_1 = {self._sim.t1},\, t_2 = {self._sim.t2},\, A_0 = {self._sim.drivingAmplitude},\, \Omega = {self._sim.drivingFreq:.5f},\, \gamma_- = {self._sim.decayConstant}$"
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
        
    def PlotTotalCurrent(self):
        """Plots the total current operator in the time and frequency domains."""

        current, fourier = self._sim.CalculateTotalCurrent()

        kValues = (np.array( list( self._sim._models.keys() ) ) / np.pi).tolist()
        title = rf"$k = {kValues} \pi,\, t_1 = {self._sim.t1},\, t_2 = {self._sim.t2},\, A_0 = {self._sim.drivingAmplitude},\, \Omega = {self._sim.drivingFreq:.5f},\, \gamma_- = {self._sim.decayConstant}$"

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
            ax[row].plot(self._sim.tAxis, self._plottingFunctions[row](current),
                        color = "Black")
    
            ax[row].set_xlabel(self._tLabel)
            ax[row].set_ylabel(yLabels[row])

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(16, 8.8))

        # Plotting the fourier transform of the current operator.
        plt.semilogy(self._sim.freqAxis / self._sim.drivingFreq, np.abs(fourier)**2,
                color = 'black')

        plt.suptitle(title)
        plt.xlabel(r"$\omega / \Omega$")
        plt.ylabel(r"$\| \tilde j (\omega) \|^2$")
        plt.show()