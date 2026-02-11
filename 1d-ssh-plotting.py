import numpy as np
from SSHModel import *

sim = SSHSimulation.Load("simulation-instances/numK: 25, numT: 10.pkl.gz")
vis = SSHVisualiser(sim)

# Everything for a fixed k.
k = sim.momentums[0]
vis.PlotSingleTimeCorrelations(k, overplotFourier=False, saveFigs=True, show=True, overplotInitialConditions=True)
# vis.PlotDoubleTimeCorrelations(k, saveFigs=True, subtractUncorrelatedValues=False, numTauPoints=None, show=False)
# vis.PlotDoubleTimeCorrelations(k, saveFigs=True, subtractUncorrelatedValues=True, numTauPoints=None, show=False)
# vis.PlotSingleTimeProducts(k, saveFigs=True, numTauPoints=None, show=False)
    
# Everything for all k.
# vis.PlotTotalCurrent(saveFig = True, show = False, overplotFourier=True)
# vis.PlotConnectedCurrentCorrelator(saveFig = True, show = False)
# vis.PlotNumericallyIntegratedHarmonics(saveFig = True, show = False, fLim=(-12.5, 12.5))
# vis.PlotIntegratedDoubleTimeCurrentCorrelation(saveFig=True, show=False)
# vis.PlotIntegratedDoubleTimeCurrentProduct(saveFig=True, overplotManualProduct=True, show=False)

# vis.PlotDoubleTimeTIntegratedCorrelations(saveFigs = True, show = False, subtractUncorrelatedValues=True)
# vis.PlotDoubleTimeTIntegratedCorrelations(saveFigs = True, show = False, subtractUncorrelatedValues=False)