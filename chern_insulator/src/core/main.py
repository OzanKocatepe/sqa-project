import numpy as np
import matplotlib.pyplot as plt

from data import EnsembleParameters
from core import Ensemble, Plotting
from hamiltonian import Hamiltonian
from config import PLOTTING_DIR

def main():
    # Total number of momentum points to sample.
    numK = 20
    tauMax = 50
    
    xAmplitude = np.zeros(4)
    yAmplitude = np.zeros(4)
    amps = [1e-4, 1e-3, 1e-2, 1e-1]
    for i, drivingAmp in enumerate(amps):
        params = EnsembleParameters(
            delta = 3,
            drivingAmp = drivingAmp,
            decayConstant = 0.2,
            maxN = 50
        )

        # Check the Chern number.
        # print(f"Trivial Phase (Delta = 3): C = {Hamiltonian.ChernNumber(3)}")
        # print(f"Non-trivial Phase (Delta = 1): C = {Hamiltonian.ChernNumber(1)}")

        ensemble = Ensemble(params)
        ensemble.SampleBrillouinZone(numK)
        # ensemble.AddMomentum((np.pi / 4, np.pi / 8))
        # ensemble.AddMomentum((np.pi / 4, -np.pi / 8))
        # ensemble.AddMomentum((-np.pi / 4, np.pi / 8))
        # ensemble.AddMomentum((-np.pi / 4, -np.pi / 8))
        ensemble.Run(tauMax)

        curr = ensemble.totalCurrent.paramagneticCurrent
        xAmplitude[i] = np.max(curr[0, -500])
        yAmplitude[i] = np.max(curr[1, -500])

    plt.plot(amps, xAmplitude, label=r'$j_x$ Amp')
    plt.plot(amps, yAmplitude, label=r'$j_y$ Amp')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Driving Amplitude")
    plt.ylabel("Steady State Amplitude")
    plt.legend()
    plt.savefig("Driving Amp vs. SS Amp.png", dpi=300)
    plt.show()

    # plot = Plotting(ensemble)
    # plot.PlotSingleTime(np.pi / 4, np.pi / 8, tMax = 20, overplotNumericalSolution=True)
    # plot.PlotSingleTime(np.pi / 4, -np.pi / 8, tMax = 10, overplotNumericalSolution=True)
    # plot.PlotSingleTime(-np.pi / 4, np.pi / 8, tMax = 10, overplotNumericalSolution=True)
    # plot.PlotSingleTime(-np.pi / 4, -np.pi / 8, tMax = 10, overplotNumericalSolution=True)
    # plot.PlotParamagneticCurrent()
    # plot.PlotParamagneticCurrentFFT(linearScale=False)

if __name__ == "__main__":
    main()