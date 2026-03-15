import numpy as np
from SSHModel import *

numK, numT = 25, 21
sim = SSHSimulation.Load(f"simulation-instances/numK: {numK}, numT: {numT}.pkl.gz")
vis = SSHVisualiser(sim, f"plots/numK: {numK}, numT: {numT}")

k = sim.momentums[2]
# for k in sim.momentums:
# vis.PlotSingleTimeCorrelations(k,
#                             saveFigs = True,
#                             show = False,
#                             overplotFourierSeries = True,
#                             overplotTAxis = True)

# vis.PlotSingleTimeCorrelationFourierDifference(k,
#                                                saveFigs = True,
#                                                show = True)

# vis.PlotDoubleTimeCorrelations(k,
#                             format = 'noise',
#                             saveFigs = True,
#                             show = False)

# vis.PlotDoubleTimeCorrelations(k,
#                             format = 'product',
#                             saveFigs = True,
#                             show = False)

# vis.PlotDoubleTimeCorrelations(k,
#                             format = 'connected',
#                             saveFigs = True,
#                             show = False)

# $\langle j(t) \rangle$
# vis.PlotCurrent(saveFigs = True,
#                 show = False,
#                 overplotFourierSeries = True
#                 )

# vis.PlotCurrentFFT(saveFigs = True,
#                    show = False
#                    )

# $\int dt\, \langle j(t) j(t + \tau) \rangle$

# vis.PlotIntegratedDoubleTimeCurrent(format='noise',
#                                     saveFigs = True,
#                                     show = False
#                                     )

# # $\int dt\, \langle j(t) \rangle \langle j(t + \tau) \rangle$

# vis.PlotIntegratedDoubleTimeCurrent(format='product',
#                                     saveFigs = True,
#                                     show = True,
#                                     overplotNumericalProduct = True
#                                     )

# vis.PlotCurrentProductDifference(saveFigs = True,
#                                  show = True
#                                  )

# # $\int dt\, \langle j(t) j (t + \tau) \rangle - \langle j(t) \rangle \langle j(t + \tau) \rangle$

vis.PlotIntegratedDoubleTimeCurrent(format='connected',
                                    saveFigs = True,
                                    # xLim = (None, 8.5),
                                    # yLim = (-2e-10, 2e-10),
                                    show = True
                                    )

# vis.PlotConnectedCurrentTerms(saveFigs = True,
#                               show = False,
#                               yLim = None
#                               )

vis.PlotConnectedCurrentFFT(saveFigs = True,
                            show = True
                            )

# vis.PlotHarmonics(saveFigs = True,
#                   show = True
#                   )