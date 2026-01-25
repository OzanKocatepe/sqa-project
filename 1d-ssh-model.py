import numpy as np
import matplotlib.pyplot as plt

from SSHSimulation import SSHSimulation
from SSHVisualiser import SSHVisualiser

tAxis = np.linspace(0, 30, 10000)
initialConditions = np.array([-0.5, -0.5, 0], dtype=complex)

sim = SSHSimulation( 
    t1 = 2,
    t2 = 1,
    decayConstant = 0.1,
    drivingAmplitude = 0.2,
    drivingFreq = 2 / 3.01
)

sim.AddMomentum([np.pi / 4, -np.pi / 4])
sim.Run(tAxis, initialConditions, steadyStateCutoff=15)

vis = SSHVisualiser(sim)
vis.PlotSingleTimeCorrelations(np.pi / 4)
vis.PlotSingleTimeCorrelations(-np.pi / 4)
vis.PlotTotalCurrent()