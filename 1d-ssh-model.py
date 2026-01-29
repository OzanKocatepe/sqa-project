import numpy as np
import matplotlib.pyplot as plt
import math

from SSHModel import *

tauAxis = np.linspace(0, 30, 200000)
numT = 1
initialConditions = np.array([-0.5, -0.5, 0], dtype=complex)

params = SSHParameters(
    t1 = 2,
    t2 = 1,
    decayConstant = 0.1,
    drivingAmplitude = 0.2,
    drivingFreq = 2 / 3.01
)

sim = SSHSimulation(params)
sim.AddMomentum(np.pi / 4)
sim.Run(tauAxis, initialConditions, numT, debug=True)

vis = SSHVisualiser(sim)
vis.PlotSingleTimeCorrelations(np.pi / 4, overplotFourier=True)
# vis.PlotDoubleTimeCorrelations(np.pi / 4, saveFigs=True, subtractUncorrelatedValues=True, numTauPoints=100)