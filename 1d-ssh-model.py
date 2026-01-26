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

# Calculates maximum allowed n based on nyquist frequency.
dx = np.mean(np.diff(tAxis / params['decayConstant']))
maxAllowedN = math.floor(1 / (4 * dx * np.pi * params['drivingFreq']))
n = maxAllowedN
print(maxAllowedN)

expectationCoeff = ssh.CalculateExpectationCoefficients(n, steadyStateCutoff=15, numPeriods=10) # (3, 2n + 1)
currentCoeff = ssh.CalculateCurrentCoefficients(n) # (3, 2n + 1)

# Stores the current expansions.
currentExpansions = np.zeros((3, tAxis.size), dtype=complex)
expectationExpansions = np.zeros((3, tAxis.size), dtype=complex)

for functionIndex in range(3):
    expectationExpansions[functionIndex, :] = ssh.EvaluateFourierExpansion(expectationCoeff[functionIndex, :], angularFreq)
    currentExpansions[functionIndex, :] = ssh.EvaluateFourierExpansion(currentCoeff[functionIndex, :], angularFreq)

current, fourier = ssh.CalculateCurrent(15)

# Comparing expectation operators.
fig, ax = plt.subplots(3, 2, figsize=(16, 8.8))
for i in range(3):
    ax[i, 0].plot(tAxis, ssh.solution.y[i].real, color='black')
    ax[i, 0].plot(tAxis, expectationExpansions[i].real, color='blue')

    ax[i, 1].plot(tAxis, ssh.solution.y[i].imag, color='black')
    ax[i, 1].plot(tAxis, expectationExpansions[i].imag, color='blue')

plt.show()

# Comparing current operators.
fig, ax = plt.subplots(2, 1, figsize=(16, 8.8))
ax[0].plot(tAxis, current.real, color='black')
# Plotting fourier expansion of current operator..
ax[0].plot(tAxis, (currentExpansions[0, :] * expectationExpansions[0]
                + currentExpansions[1, :] * expectationExpansions[1]
                + currentExpansions[2, :] * expectationExpansions[2]).real, color='blue')

ax[1].plot(tAxis, current.imag, color='black')
# Plotting fourier expansion of current operator..
ax[1].plot(tAxis, (currentExpansions[0, :] * expectationExpansions[0]
                + currentExpansions[1, :] * expectationExpansions[1]
                + currentExpansions[2, :] * expectationExpansions[2]).imag, color='blue')

plt.show()