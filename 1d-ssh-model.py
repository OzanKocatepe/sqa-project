import numpy as np
from SSHModel import *

# k = np.pi / 4
if __name__ == "__main__":
    # numK = 25
    # numT = 10
    # tauAxis = np.linspace(0, 100, 40000)
    # steadyStateCutoff = 60

    numK = 5
    numT = 3
    tauAxis = np.linspace(0, 60, 40000)
    steadyStateCutoff = 40

    initialConditions = np.array([-0.5, -0.5, 0], dtype=complex)

    params = data.EnsembleParameters(
        t1 = 2,
        t2 = 1,
        decayConstant = 0.1,
        drivingAmplitude = 0.2, # 0.2
        drivingFreq = 2 / 3.01,
        maxN = 11
    )

    sim = SSHSimulation(params)
    sim.AddMomentum(np.linspace(-np.pi, np.pi, numK))
    # sim.AddMomentum(np.pi / 4)
    sim.Run(
        initialConditions = initialConditions,
        tauAxisDim = tauAxis,
        steadyStateCutoff = steadyStateCutoff,
        numT = numT,
        numProcesses = 5
    )

    sim.Save("simulation-instances")
    # sim.ExportAllRecords('simulation-instances')
    # sim.ExportRecordSummary('simulation-instances')