import numpy as np
import matplotlib.pyplot as plt
import cProfile
import argparse

from data import EnsembleParameters
from core.Ensemble import Ensemble
from core.Plotting import Plotting
from Topology import ChernNumber
from config.paths import PLOTTING_DIR, STYLESHEET, DATA_DIR

def main() -> None:
    # Create the parser.
    parser = argparse.ArgumentParser(
        description = "Solves for the first- and second- order correlation functions for " \
        "a two-dimensional Chern insulator."
    )
    
    # Define the arguments.
    parser.add_argument(
        "-k",
        "--numK",
        help = "The number of momentum points on each axis to sample the Brillouin Zone with.",
        type = int,
        required = True
    )
    parser.add_argument(
        "-t",
        "--numT",
        help = "The number of initial conditions within one driving period to use for the" \
        "second-order correlation functions.",
        type = int,
        required = True
    )
    parser.add_argument(
        "-c",
        "--cores",
        help = "The number of cores to use. The input will be sanitized.",
        type = int,
        required = True
    )
    parser.add_argument(
        "-s",
        "--save",
        help = "Tells the script to save the total current object to disk." \
        " Saves to data/, which will be created if it doesn't already exist.",
        action = "store_true"
    )

    # Parse the arguments.
    args = parser.parse_args()

    # Defines our values.
    numK = args.numK
    numT = args.numT
    numProcesses = args.cores
    tauMax = 20

    # Check the Chern number.
    # print(f"Trivial Phase (Delta = 3): C = {ChernNumber(3)}")
    # print(f"Non-trivial Phase (Delta = 1): C = {ChernNumber(1)}")

    plt.style.use(STYLESHEET)

    params = EnsembleParameters(
        delta = 1,
        drivingAmp = 0.2,
        decayConstant = 0.2,
        maxN = 50
    )

    ensemble = Ensemble(params)
    ensemble.SampleBrillouinZone(numK)
    # ensemble.AddMomentum((np.pi / 4, np.pi / 8))
    # ensemble.AddMomentum((np.pi / 4, -np.pi / 8))
    # ensemble.AddMomentum((-np.pi / 4, np.pi / 8))
    # ensemble.AddMomentum((-np.pi / 4, -np.pi / 8))
    ensemble.Run(tauMax, numT, numProcesses=numProcesses)
    if args.save:
        ensemble.SaveCurrent()

    # plot = Plotting(ensemble)
    # plot.PlotSingleTime(np.pi / 4, np.pi / 8, tMax = 20, overplotNumericalSolution=True)
    # plot.PlotSingleTime(np.pi / 4, -np.pi / 8, tMax = 10, overplotNumericalSolution=True)
    # plot.PlotSingleTime(-np.pi / 4, np.pi / 8, tMax = 10, overplotNumericalSolution=True)
    # plot.PlotSingleTime(-np.pi / 4, -np.pi / 8, tMax = 10, overplotNumericalSolution=True)
    # plot.PlotDoubleTimeCorrelation(np.pi / 4, np.pi / 8)
    # plot.PlotDoubleTimeCurrent()
    # plot.PlotTotalCurrent(overplotLengthGauge=False)
    # plot.PlotTotalCurrentFFT(linearScale=False, overplotLengthGauge=False)

if __name__ == "__main__":
    # Runs the main function.
    main()