import numpy as np
import matplotlib.pyplot as plt

from data import EnsembleParameters
from core import Ensemble, Plotting
from hamiltonian import Hamiltonian
from config.paths import PLOTTING_DIR, STYLESHEET

def main():
    # Total number of momentum points to sample.
    numK = 10
    tauMax = 50

    plt.style.use(STYLESHEET)

    deltas = np.arange(0.5, 1.5 + 0.5, 0.5, dtype=float)
    xAmps = np.zeros_like(deltas, dtype=float)
    yAmps = np.zeros_like(deltas, dtype=float)

    for i in range(deltas.size):
        params = EnsembleParameters(
            delta = deltas[i],
            drivingAmp = 0.2,
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

        totalCurrent = ensemble.totalCurrent
        xAmps[i] = np.max(np.abs(totalCurrent.paramagneticCurrent[0, :]))
        yAmps[i] = np.max(np.abs(totalCurrent.paramagneticCurrent[1, :]))

        # plot = Plotting(ensemble)
        # plot.PlotSingleTime(np.pi / 4, np.pi / 8, tMax = 20, overplotNumericalSolution=True)
        # plot.PlotSingleTime(np.pi / 4, -np.pi / 8, tMax = 10, overplotNumericalSolution=True)
        # plot.PlotSingleTime(-np.pi / 4, np.pi / 8, tMax = 10, overplotNumericalSolution=True)
        # plot.PlotSingleTime(-np.pi / 4, -np.pi / 8, tMax = 10, overplotNumericalSolution=True)
        # plot.PlotParamagneticCurrent()
        # plot.PlotParamagneticCurrentFFT(linearScale=False)

    np.save(PLOTTING_DIR / "Current Amplitude vs Delta.npy", np.stack([xAmps, yAmps], axis=0))

    plt.plot(deltas, xAmps, '-x', label=r'$j_x$ Amplitude', color='blue')
    plt.plot(deltas, yAmps, '-x', label=r'$j_y$ Amplitude', color='orange')
    plt.xlabel(r'$\Delta$')
    plt.ylabel("Amplitude of Total Current")
    plt.suptitle(fr"{numK} x {numK} Momentums in BZ, $A_0$ = {0.2}, $\gamma$ = {0.2}")
    plt.legend()
    plt.savefig(PLOTTING_DIR / "Current Amplitude vs Delta.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()