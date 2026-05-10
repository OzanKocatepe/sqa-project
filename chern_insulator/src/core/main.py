import numpy as np
import matplotlib.pyplot as plt

from data import EnsembleParameters
from core.Ensemble import Ensemble
from core.Plotting import Plotting
from Topology import ChernNumber
from config.paths import PLOTTING_DIR, STYLESHEET

def main():
    # Total number of momentum points to sample.
    numK = 3
    numT = 5
    tauMax = 20

    # Check the Chern number.
    # print(f"Trivial Phase (Delta = 3): C = {ChernNumber(3)}")
    # print(f"Non-trivial Phase (Delta = 1): C = {ChernNumber(1)}")

    plt.style.use(STYLESHEET)

    params = EnsembleParameters(
        delta = 3,
        drivingAmp = 0.2,
        decayConstant = 0.2,
        maxN = 50
    )

    ensemble = Ensemble(params)
    # ensemble.SampleBrillouinZone(numK)
    ensemble.AddMomentum((np.pi / 4, np.pi / 8))
    # ensemble.AddMomentum((np.pi / 4, -np.pi / 8))
    # ensemble.AddMomentum((-np.pi / 4, np.pi / 8))
    # ensemble.AddMomentum((-np.pi / 4, -np.pi / 8))
    ensemble.Run(tauMax, numT, numProcesses=1)

    plot = Plotting(ensemble)
    # plot.PlotSingleTime(np.pi / 4, np.pi / 8, tMax = 20, overplotNumericalSolution=True)
    # plot.PlotSingleTime(np.pi / 4, -np.pi / 8, tMax = 10, overplotNumericalSolution=True)
    # plot.PlotSingleTime(-np.pi / 4, np.pi / 8, tMax = 10, overplotNumericalSolution=True)
    # plot.PlotSingleTime(-np.pi / 4, -np.pi / 8, tMax = 10, overplotNumericalSolution=True)
    # plot.PlotDoubleTimeCorrelation(np.pi / 4, np.pi / 8)
    plot.PlotDoubleTimeCurrent()
    # plot.PlotTotalCurrent(overplotLengthGauge=False)
    # plot.PlotTotalCurrentFFT(linearScale=False, overplotLengthGauge=False)

if __name__ == "__main__":
    main()