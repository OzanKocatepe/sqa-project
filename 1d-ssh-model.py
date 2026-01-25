import numpy as np
import matplotlib.pyplot as plt

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

ssh = SSH(k = np.pi / 4, **params)
ssh.Solve(tAxis, initialConditions)
n = 10
coefficients = ssh.CalculateFourierCoefficients(n) # (3, 2n + 1)

exponentialTerms = np.zeros((2 * n + 1, tAxis.size), dtype=complex) # (2n + 1, tAxis.size)
for i in np.arange(-n, n + 1):
    exponentialTerms[i + n, :] = np.exp(2j * np.pi * i * params['drivingFreq'] * tAxis / params['decayConstant'])

fourierExpansions = np.zeros((3, tAxis.size), dtype=complex)
for functionIndex in range(3):
    for t in range(tAxis.size):
        fourierExpansions[functionIndex, t] = np.dot(coefficients[functionIndex, :], exponentialTerms[:, t])





# expectationLabels = [r"$\langle \tilde \sigma_-(t) \rangle$",
#                      r"$\langle \tilde \sigma_+(t) \rangle$",
#                      r"$\langle \tilde \sigma_z(t) \rangle$"]

expectationLabels = ['-', '+', 'z']
                
yLabels = []
for i in range(len(expectationLabels)):
    yLabels.append(
        [f"Magnitude of {expectationLabels[i]}",
         f"Real Part of {expectationLabels[i]}",
         f"Imaginary Part of {expectationLabels[i]}"]
    )

plottingFunctions = [lambda z: np.abs(z), lambda z: z.real, lambda z: z.imag]
tLabel = r"$t \gamma_-$"

# Creates the 3x3 subplots.
nrows, ncols = 3, 3
fig, ax = plt.subplots(nrows, ncols, figsize=(16, 8.8))

for row in np.arange(nrows):
    for col in np.arange(ncols):
        # Plot numerical solution.
        ax[row, col].plot(tAxis, plottingFunctions[col](ssh.solution.y[row]),
                        color = "black")
        # Plots fourier expansion.
        ax[row, col].plot(tAxis, plottingFunctions[col](fourierExpansions[row, :]),
                        color = "Blue")
        
        # Sets other properties.
        ax[row, col].set_xlabel(tLabel)
        ax[row, col].set_ylabel(yLabels[row][col])

plt.tight_layout()
plt.show()