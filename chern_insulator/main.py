import numpy as np
import matplotlib.pyplot as plt

from data import EnsembleParameters
from Ensemble import Ensemble
from Plotting import Plotting

# Number of TOTAL momentum points.
numK = 8**2
tauMax = 50

params = EnsembleParameters(
    delta = 3,
    drivingAmp = 0.2,
    decayConstant = 0.2,
    maxN = 50
)

ensemble = Ensemble(params)
# ensemble.SampleBrillouinZone(numK)
ensemble.AddMomentum((np.pi / 4, np.pi / 8))
ensemble.AddMomentum((np.pi / 4, -np.pi / 8))
ensemble.AddMomentum((-np.pi / 4, np.pi / 8))
ensemble.AddMomentum((-np.pi / 4, -np.pi / 8))
ensemble.Run(tauMax)

plot = Plotting(ensemble)
# plot.PlotSingleTime(np.pi / 4, np.pi / 8, tMax = 10, overplotNumericalSolution=False)
plot.PlotParamagneticCurrent()
# plot.PlotParamagneticCurrentFFT(linearScale=False)