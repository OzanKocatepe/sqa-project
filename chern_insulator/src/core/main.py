import numpy as np
import matplotlib.pyplot as plt

from data import EnsembleParameters
from core import Ensemble, Plotting
from hamiltonian import Hamiltonian
from config.paths import PLOTTING_DIR, STYLESHEET

def main():
    # Total number of momentum points to sample.
    numK = 20
    tauMax = 50

    plt.style.use(STYLESHEET)

    amps = np.linspace(1e-4, 1e-1, 10)
    angularFreqs = np.zeros(len(amps), dtype=float)
    xAmps = np.zeros(len(amps), dtype=float)
    yAmps = np.zeros(len(amps), dtype=float)
    xLGAmps = np.zeros(len(amps), dtype=float)
    yLGAmps = np.zeros(len(amps), dtype=float)

    for i in range(len(amps)):
        params = EnsembleParameters(
            delta = 3,
            drivingAmp = amps[i],
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

        angularFreqs[i] = params.angularFreq
        xAmps[i] = np.max(np.abs(ensemble.totalCurrent.paramagneticCurrent[0, -1000]))
        yAmps[i] = np.max(np.abs(ensemble.totalCurrent.paramagneticCurrent[1, -1000]))
        xLGAmps[i] = np.max(np.abs(ensemble.totalCurrent.lengthGaugeCurrent[0, -1000]))
        yLGAmps[i] = np.max(np.abs(ensemble.totalCurrent.lengthGaugeCurrent[1, -1000]))

    # plot = Plotting(ensemble)
    # plot.PlotSingleTime(np.pi / 4, np.pi / 8, tMax = 20, overplotNumericalSolution=True)
    # plot.PlotSingleTime(np.pi / 4, -np.pi / 8, tMax = 10, overplotNumericalSolution=True)
    # plot.PlotSingleTime(-np.pi / 4, np.pi / 8, tMax = 10, overplotNumericalSolution=True)
    # plot.PlotSingleTime(-np.pi / 4, -np.pi / 8, tMax = 10, overplotNumericalSolution=True)
    # plot.PlotParamagneticCurrent(overplotLengthGauge=True)
    # plot.PlotParamagneticCurrentFFT(linearScale=False, overplotLengthGauge=True)

    plt.plot(amps * angularFreqs, xAmps, '-x', color='tab:blue', label=r"$j_x$")
    plt.plot(amps * angularFreqs, yAmps, '-x', color='orange', label=r"$j_y$")
    plt.plot(amps * angularFreqs, xLGAmps, '-x', color='purple', label=r"$j_x$ (LG)")
    plt.plot(amps * angularFreqs, yLGAmps, '-x', color='red', label=r"$j_y$ (LG)")
    plt.xlabel(r"Electric Field Amplitude ($A_0 \omega$)")
    plt.ylabel("Current Amplitude")
    plt.legend()
    plt.savefig(f"{PLOTTING_DIR}/Delta {params.delta}/LG Current Amp vs E Field", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()