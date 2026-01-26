import numpy as np
import matplotlib.pyplot as plt
import math

from SSH import SSH
from SSHSimulation import SSHSimulation
from SSHVisualiser import SSHVisualiser

tAxis = np.linspace(0, 30, 10000)
initialConditions = np.array([-0.5, -0.5, 0], dtype=complex)

params = {
    't1' : 2,
    't2' : 1,
    'decayConstant' : 0.1,
    'drivingAmplitude' : 0.2,
    'drivingFreq' : 2 / 3.01
}
angularFreq = 2 * np.pi * params['drivingFreq']

ssh = SSH(k = np.pi / 4, **params)
ssh.Solve(tAxis, initialConditions)

coeff = ssh.CalculateCurrentExpectationCoefficients()
expansion = ssh.EvaluateFourierExpansion(coeff)
current, fourier = ssh.CalculateCurrent(15)

# Comparing current operators.
fig, ax = plt.subplots(2, 1, figsize=(16, 8.8))
ax[0].plot(tAxis, current.real, color='black')
ax[0].plot(tAxis, expansion.real, color='blue')

ax[1].plot(tAxis, current.imag, color='black')
ax[1].plot(tAxis, expansion.imag, color='blue')

plt.show()