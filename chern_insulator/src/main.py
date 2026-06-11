import numpy as np
import matplotlib.pyplot as plt
import cProfile
import argparse
import psutil
import os
from datetime import datetime
import threading

from data import EnsembleParameters
from core import Ensemble, Plotting
from topology import chern_number
from config.paths import PLOTTING_DIR, STYLESHEET, DATA_DIR

def log_memory(interval_seconds=300, log_file=DATA_DIR / "memory_log.txt"):
    process = psutil.Process(os.getpid())

    while True:
        mem = process.memory_info().rss

        for child in process.children(recursive=True):
            try:
                mem += child.memory_info().rss
            except psutil.NoSuchProcess:
                pass

        ram_mb = mem / 1024 ** 2
        timestamp = datetime.now().strftime("%H:%M:%S")
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(log_file, 'a') as f:
            f.write(f"{timestamp} - {ram_mb:.1f} MB\n")
        threading.Event().wait(interval_seconds)

def main() -> None:
    # Create the parser.
    parser = argparse.ArgumentParser(
        description = "Solves for the first- and second- order correlation functions for " \
        "a two-dimensional Chern insulator."
    )
    
    # Define the arguments.
    parser.add_argument(
        "-d",
        "--delta",
        help = "The delta parameter.",
        type = float,
        required = True
    )
    parser.add_argument(
        "-k",
        "--numK",
        help = "The number of momentum points on each axis to sample the Brillouin Zone with.",
        type = int,
        default = 61
    )
    parser.add_argument(
        "-t",
        "--numT",
        help = "The number of initial conditions within one driving period to use for the" \
        "second-order correlation functions.",
        type = int,
        default = 21
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
    parser.add_argument(
        "--log-memory",
        help = "Whether to log the memory usage to disk in 5 minute intervals.",
        action = "store_true",
    )
    parser.add_argument(
        "--disable-second-order",
        help = "Disables the second-order calculations, heavily speeding up the program.",
        action = "store_true"
    )

    # Parse the arguments.
    args = parser.parse_args()

    # Defines our values.
    numK = args.numK
    numT = args.numT
    numProcesses = args.cores
    delta = args.delta
    save_flag = args.save
    log_memory_flag = args.log_memory
    disable_second_order = args.disable_second_order

    if log_memory_flag:
        # Start the logging thread.
        t = threading.Thread(target=log_memory, daemon=True)
        t.start()

    # Check the Chern number.
    # print(f"Trivial Phase (Delta = 3): C = {chern_number(3)}")
    # print(f"Non-trivial Phase (Delta = 1): C = {chern_number(1)}")

    plt.style.use(STYLESHEET)

    params = EnsembleParameters(
        delta = delta,
        drivingAmp = 0,
        decayConstant = 0.2,
        maxN = 50,
        num_particles = 10e6
    )
    tauMax = 20

    ensemble = Ensemble(params)
    ensemble.SampleBrillouinZone(numK)
    # ensemble.AddMomentum((np.pi / 4, np.pi / 8))
    # ensemble.AddMomentum((np.pi / 4, -np.pi / 8))
    # ensemble.AddMomentum((-np.pi / 4, np.pi / 8))
    # ensemble.AddMomentum((-np.pi / 4, -np.pi / 8))
    ensemble.Run(tauMax, numT, numProcesses=numProcesses, disable_second_order=disable_second_order)
    if save_flag:
        ensemble.SaveCurrent()

    plot = Plotting(ensemble)
    # plot.PlotSingleTime(np.pi / 4, np.pi / 8, tMax = 20, overplotNumericalSolution=True)
    # plot.PlotSingleTime(np.pi / 4, -np.pi / 8, tMax = 10, overplotNumericalSolution=True)
    # plot.PlotSingleTime(-np.pi / 4, np.pi / 8, tMax = 10, overplotNumericalSolution=True)
    # plot.PlotSingleTime(-np.pi / 4, -np.pi / 8, tMax = 10, overplotNumericalSolution=True)
    # plot.PlotDoubleTimeCorrelation(np.pi / 4, np.pi / 8)
    # plot.PlotDoubleTimeCurrent()
    # plot.PlotTotalCurrent(overplotLengthGauge=False)
    # plot.PlotTotalCurrentFFT(linearScale=False, overplotLengthGauge=False)
    # plot.PlotIntegratedSecondOrderCurrent()

if __name__ == "__main__":
    main()