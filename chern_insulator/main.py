import numpy as np
import matplotlib.pyplot as plt

from data import EnsembleParameters

# Number of TOTAL momentum points.
numK = 75

params = EnsembleParameters(
    delta = 1,
    drivingAmp = 0.3,
    drivingFreq = 2 / 3.01 * 1 / (2 * np.pi),
    decayConstant = 0.2,
    maxN = 50
)