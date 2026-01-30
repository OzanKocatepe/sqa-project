import numpy as np
import matplotlib.pyplot as plt
import math

from SSHModel import *

tauAxis = np.linspace(0, 30, 200)
initialConditions = np.array([-0.5, -0.5, 0], dtype=complex)

params = SSHParameters(
    t1 = 2,
    t2 = 1,
    decayConstant = 0.1,
    drivingAmplitude = 0, # 0.2
    drivingFreq = 2 / 3.01
)

k = np.pi / 4
numT = 5
sim = SSHSimulation(params)
sim.AddMomentum(k)
sim.Run(tauAxis, initialConditions, numT, debug=True)

vis = SSHVisualiser(sim)
vis.PlotSingleTimeCorrelations(k, overplotFourier=True)
vis.PlotDoubleTimeCorrelations(k, saveFigs=False, subtractUncorrelatedValues=False, numTauPoints=None)