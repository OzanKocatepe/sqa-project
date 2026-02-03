import numpy as np
import matplotlib.pyplot as plt
import math

from SSHModel import *

# k = np.pi / 4
if __name__ == "__main__":
    numK = 25
    numT = 10
    tauAxis = np.linspace(0, 30, 20000)
    initialConditions = np.array([-0.5, -0.5, 0], dtype=complex)

    params = EnsembleParameters(
        t1 = 2,
        t2 = 1,
        decayConstant = 0.1,
        drivingAmplitude = 0.2, # 0.2
        drivingFreq = 2 / 3.01
    )

    sim = SSHSimulation(params)
    sim.AddMomentum(np.linspace(-np.pi, np.pi, numK))
    # sim.AddMomentum(np.pi / 4)
    sim.Run(tauAxis, initialConditions, numT, debug=True)

    vis = SSHVisualiser(sim)
    # vis.PlotSingleTimeCorrelations(k, overplotFourier=False, saveFigs=True, show=False)
    # vis.PlotDoubleTimeCorrelations(k, saveFigs=True, subtractUncorrelatedValues=False, numTauPoints=None, show=False)
    # vis.PlotDoubleTimeCorrelations(k, saveFigs=True, subtractUncorrelatedValues=True, numTauPoints=None, show=False)
    # vis.PlotSingleTimeProducts(k, saveFigs=True, numTauPoints=None, show=False)
    # vis.PlotTotalCurrent(saveFig = True, show = False)
    # vis.PlotConnectedCurrentCorrelator(saveFig = True, show = False)
    # vis.PlotIntegratedDoubleTimeCurrentCorrelation(saveFig=True, show=False)
    # vis.PlotIntegratedDoubleTimeCurrentProduct(saveFig=True, show=False)