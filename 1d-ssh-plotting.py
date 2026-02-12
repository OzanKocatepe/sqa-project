import numpy as np
from SSHModel import *

sim = SSHSimulation.Load("simulation-instances/numK: 25, numT: 10.pkl.gz")
vis = SSHVisualiser(sim, "plots/numK: 25, numT: 10")

# $\langle j(t) \rangle$.
# vis.PlotCurrent(saveFigs = True,
#                 show = True,
#                 overplotFourierSeries = True
#                 )

# vis.PlotCurrentFFT(saveFigs = True,
#                    show = True
#                    )

# $\int dt\, \langle j(t) j(t + \tau) \rangle$

# vis.PlotIntegratedDoubleTimeCurrent(format='noise',
#                                     saveFigs = True,
#                                     show = True
#                                     )

# $\int dt\, \langle j(t) \rangle \langle j(t + \tau) \rangle$

vis.PlotIntegratedDoubleTimeCurrent(format='product',
                                    saveFigs = True,
                                    show = True,
                                    overplotNumericalProduct = True
                                    )

# $\int dt\, \langle j(t) j (t + \tau) \rangle - \langle j(t) \rangle \langle j(t + \tau) \rangle$

# vis.PlotIntegratedDoubleTimeCurrent(format='connected',
#                                     saveFigs = True,
#                                     show = True
#                                     )

# vis.PlotConnectedCurrentFFT(saveFigs = True,
#                             show = True)

# vis.PlotHarmonics(saveFigs = True,
#                   show = True)