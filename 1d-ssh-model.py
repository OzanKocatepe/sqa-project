import numpy as np
from SSHModel import *

# k = np.pi / 4
if __name__ == "__main__":
    numK = 25
    numT = 10
    tauAxis = np.linspace(0, 100, 40000)
    initialConditions = np.array([-0.5, -0.5, 0], dtype=complex)

    params = data.EnsembleParameters(
        t1 = 2,
        t2 = 1,
        decayConstant = 0.1,
        drivingAmplitude = 0.2, # 0.2
        drivingFreq = 2 / 3.01
    )

    sim = SSHSimulation(params)
    sim.AddMomentum(np.linspace(-np.pi, np.pi, numK))
    # sim.AddMomentum(np.pi / 4)
    sim.Run(
        initialConditions = initialConditions,
        tauAxisDim = tauAxis,
        steadyStateCutoff = 60,
        numT = numT,
        numProcesses = 6
    )

    sim.Save("simulation-instances")