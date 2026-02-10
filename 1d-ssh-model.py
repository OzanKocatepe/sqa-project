import numpy as np
import matplotlib.pyplot as plt
import math

from SSHModel import *

# k = np.pi / 4
if __name__ == "__main__":
    numK = 11
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
        numProcesses = 6)

    # vis = SSHVisualiser(sim)

    # Everything for a fixed k.
    # momentums = sim.momentums
    # for k in [momentums[3], momentums[-4]]:
    #     vis.PlotSingleTimeCorrelations(k, overplotFourier=False, saveFigs=True, show=True, overplotInitialConditions=True)
    #     vis.PlotDoubleTimeCorrelations(k, saveFigs=True, subtractUncorrelatedValues=False, numTauPoints=None, show=False)
    #     vis.PlotDoubleTimeCorrelations(k, saveFigs=True, subtractUncorrelatedValues=True, numTauPoints=None, show=False)
    #     vis.PlotSingleTimeProducts(k, saveFigs=True, numTauPoints=None, show=False)
    
    # Everything for all k.
    # vis.PlotTotalCurrent(saveFig = True, show = False, overplotFourier=True)
    # vis.PlotConnectedCurrentCorrelator(saveFig = True, show = False)
    # vis.PlotNumericallyIntegratedHarmonics(saveFig = True, show = False, fLim=(-12.5, 12.5))
    # vis.PlotIntegratedDoubleTimeCurrentCorrelation(saveFig=True, show=False)
    # vis.PlotIntegratedDoubleTimeCurrentProduct(saveFig=True, overplotManualProduct=True, show=False)

    # vis.PlotDoubleTimeTIntegratedCorrelations(saveFigs = True, show = False, subtractUncorrelatedValues=True)
    # vis.PlotDoubleTimeTIntegratedCorrelations(saveFigs = True, show = False, subtractUncorrelatedValues=False)