import numpy as np
import matplotlib.pyplot as plt
import math

from SSHModel import *

tauAxis = np.linspace(0, 30, 20000)
initialConditions = np.array([-0.5, -0.5, 0], dtype=complex)

params = SSHParameters(
    t1 = 2,
    t2 = 1,
    decayConstant = 0.1,
    drivingAmplitude = 0.2, # 0.2
    drivingFreq = 2 / 3.01
)

k = np.pi / 4
numT = 3
sim = SSHSimulation(params)
sim.AddMomentum(k)
sim.Run(tauAxis, initialConditions, numT, debug=True)

vis = SSHVisualiser(sim)
# vis.PlotSingleTimeCorrelations(k, overplotFourier=True)
# vis.PlotDoubleTimeCorrelations(k, saveFigs=False, subtractUncorrelatedValues=True, numTauPoints=None, vLim=(-1.5, 1.5))
vis.PlotSingleTimeProducts(k, saveFigs=False, numTauPoints=None, slice=[(0, 0)], vLim=(-1.5, 1.5))