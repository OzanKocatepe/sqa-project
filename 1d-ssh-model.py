import numpy as np
import sys

from SSHModel import *

# k = np.pi / 4
if __name__ == "__main__":
    # Parses the input.
    if len(sys.argv) > 1:
        mode = sys.argv[1]

        match mode:
            case "full":
                numK = 75
                numT = 10
                tauAxis = np.linspace(0, 100, 40000)
                steadyStateCutoff = 60

            case "med":
                numK = 25
                numT = 10
                tauAxis = np.linspace(0, 100, 40000)
                steadyStateCutoff = 60

            case "min":
                numK = 5
                numT = 3
                tauAxis = np.linspace(0, 60, 40000)
                steadyStateCutoff = 40

            case _:
                print("Invalid mode.")
                quit()

    # initialConditions = np.array([-0.5, -0.5, 0], dtype=complex)
    initialConditions = np.array([0, 0, -1], dtype=complex)

    params = data.EnsembleParameters(
        t1 = 2,
        t2 = 1,
        decayConstant = 0.1,
        drivingAmplitude = 0.2, # 0.2
        drivingFreq = 2 / 3.01,
        maxN = 50
    )

    sim = SSHSimulation(params)
    sim.AddMomentum(np.linspace(-np.pi, np.pi, 25))

    # sim.AddMomentum(np.pi / 4)
    numT = 21
    tauAxis = np.linspace(0, 100, 5000)
    steadyStateCutoff = 60

    sim.Run(
        initialConditions = initialConditions,
        tauAxisDim = tauAxis,
        steadyStateCutoff = steadyStateCutoff,
        numT = numT,
        numProcesses = 6
    )

    sim.Save("simulation-instances")
    # sim.ExportAllRecords('simulation-instances')
    sim.ExportRecordSummary('simulation-instances')