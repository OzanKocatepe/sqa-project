import numpy as np
import matplotlib.pyplot as plt

from data import EnsembleParameters
from core import Ensemble, Plotting
from hamiltonian import Hamiltonian

def main():
    # Total number of momentum points to sample.
    numK = 10
    tauMax = 50

    params = EnsembleParameters(
        delta = 1,
        drivingAmp = 0.2,
        decayConstant = 0.2,
        maxN = 50
    )

    # Check the Chern number.
    print(f"Trivial Phase (Delta = 3): C = {Hamiltonian.ChernNumber(3)}")
    print(f"Non-trivial Phase (Delta = 1): C = {Hamiltonian.ChernNumber(1)}")

    ensemble = Ensemble(params)
    ensemble.SampleBrillouinZone(numK)
    # ensemble.AddMomentum((np.pi / 4, np.pi / 8))
    # ensemble.AddMomentum((np.pi / 4, -np.pi / 8))
    # ensemble.AddMomentum((-np.pi / 4, np.pi / 8))
    # ensemble.AddMomentum((-np.pi / 4, -np.pi / 8))
    ensemble.Run(tauMax)

    plot = Plotting(ensemble)
    # plot.PlotSingleTime(np.pi / 4, np.pi / 8, tMax = 10, overplotNumericalSolution=True)
    # plot.PlotSingleTime(np.pi / 4, -np.pi / 8, tMax = 10, overplotNumericalSolution=True)
    # plot.PlotSingleTime(-np.pi / 4, np.pi / 8, tMax = 10, overplotNumericalSolution=True)
    # plot.PlotSingleTime(-np.pi / 4, -np.pi / 8, tMax = 10, overplotNumericalSolution=True)
    plot.PlotParamagneticCurrent()
    plot.PlotParamagneticCurrentFFT(linearScale=False)

if __name__ == "__main__":
    main()