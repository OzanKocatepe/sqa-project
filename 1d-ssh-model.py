import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from typing import Callable, Any

from SSH import SSH

tAxis = np.linspace(0, 30, 250)
initialConditions = np.array([0, 0, -1], dtype=complex)

# Initialises the model with the desired values.
ssh = SSH(
    k = np.pi / 4,
    t1 = 2,
    t2 = 1 + 0j,
    decayConstant = 0.1,
    drivingAmplitude = 0.2,
    drivingFreq = 2 / 3.01
)

# Calculates the numerical solution.
numericalSol = ssh.CalculateSingleTimeCorrelations(tAxis, initialConditions, ssh.ClassicalDrivingTerm)
# Calculates the current operator.
currentOperator, currentOperatorFourier = ssh.CalculateCurrentOperator()


# ===========================================
# ==== PLOTTING SINGLE-TIME CORRELATIONS ====
# ===========================================

# Writes the labels for each correlation that we are plotting.
correlationLabels = [r"$\langle \tilde \sigma_-(t) \rangle$",
                     r"$\langle \tilde \sigma_+(t) \rangle$",
                     r"$\langle \tilde \sigma_z(t) \rangle$"]

# Title of the plot.
title = rf"$t_1 = {ssh.t1},\, t_2 = {ssh.t2},\, A_0 = {ssh.drivingAmplitude},\, \Omega = {ssh.drivingFreq:.5f},\, k = {ssh.k / np.pi} \pi,\, \gamma_- = {ssh.decayConstant}$"
xLabel = r"$t / \gamma_-$"

# Writes the labels for each individal subplot.
yLabels = []
for i in range(len(correlationLabels)):
    yLabels.append(
        [f"Magnitude of {correlationLabels[i]}",
        f"Real Part of {correlationLabels[i]}",
        f"Imaginary Part of {correlationLabels[i]}"]
    )

# The functions that we will be applying to the correlation functions.
plottingFunctions = [lambda z: np.abs(z), lambda z: z.real, lambda z: z.imag]

nrows, ncols = 3, 3
fig, ax = plt.subplots(nrows, ncols, figsize=(16, 8.8))

for row in np.arange(nrows):
    for col in np.arange(ncols):
        # Plot numerical solution.
        ax[row, col].plot(tAxis, plottingFunctions[col](numericalSol.y[row]),
                        color = "Black")
        
        # Sets other properties.
        ax[row, col].set_xlabel(xLabel)
        ax[row, col].set_ylabel(yLabels[row][col])

plt.suptitle(title)
plt.tight_layout()
plt.show()

# ===================================
# ==== PLOTTING CURRENT OPERATOR ====
# ===================================

currentLabel = r"$\langle\tilde j_k \rangle$"
yLabels = [
    f"Magnitude of {currentLabel}",
    f"Real Part of {currentLabel}",
    f"Imaginary Part of {currentLabel}"
]

nrows, ncols = 3, 1
fig, ax = plt.subplots(nrows, ncols, figsize=(16, 8.8))

for row in np.arange(nrows):
    # Plot the numerical solution.
    ax[row].plot(tAxis, plottingFunctions[row](currentOperator),
                color = "Black")
    
    ax[row].set_xlabel(xLabel)
    ax[row].set_ylabel(yLabels[row])

plt.suptitle(title)
plt.tight_layout()
plt.show()

# Plotting the fourier transform of the current operator.
freqAxis = ssh.GetFrequencyAxis()
plt.plot(freqAxis / (ssh.drivingFreq), np.abs(currentOperatorFourier)**2,
        color = 'black')

plt.suptitle(title)
# plt.xlim(-2.5, 2.5)
plt.xlabel(r"$\omega / \Omega$")
plt.ylabel(r"$\| \tilde j (\omega) \|^2$")
plt.show()