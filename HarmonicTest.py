import numpy as np
import matplotlib.pyplot as plt
from SSHModel import SSH, ModelParameters, CorrelationData, CurrentData

# Defines the irrelevant parameters to be zero.
params = ModelParameters(
    k = 0,
    t1 = 0,
    t2 = 0,
    decayConstant = 0,
    drivingAmplitude = 0,
    drivingFreq = 2
)

frequencies = np.array([1, 3, 5]) * params.drivingFreq
tauAxis = np.linspace(0, 30, 10000)
signal = np.sum( np.sin( np.outer(2 * np.pi * frequencies, tauAxis) ), axis = 0) * 1e-4

# plt.plot(tauAxis, signal)
# plt.show()

# Sets up required values.
ssh = SSH(params)
ssh._SSH__correlationData = CorrelationData(
    tauAxisSec = tauAxis
)
ssh._SSH__currentData = CurrentData(
    doubleConnectedCorrelator = signal
)

# Call function to get harmonics of signal along the entire tauAxis.
ssh._SSH__ManuallyCalculateFourierAtHarmonics(np.arange(tauAxis.size))
harmonics = ssh._SSH__currentData.harmonics

plt.semilogy(np.arange(-12, 13), np.abs(harmonics)**2, 'o')
plt.xlim(-12.5, 12.5)
plt.show()