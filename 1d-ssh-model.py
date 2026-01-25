import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from typing import Callable, Any

from SSHSimulation import SSHSimulation

tAxis = np.linspace(0, 30, 10000)
initialConditions = np.array([-0.5, -0.5, 0], dtype=complex)

simulation = SSHSimulation( 
    t1 = 2,
    t2 = 1,
    decayConstant = 0.1,
    drivingAmplitude = 0.2,
    drivingFreq = 2 / 3.01
)

simulation.AddMomentum([np.pi / 4, -np.pi / 4])
simulation.Run(tAxis, initialConditions, steadyStateCutoff=15)
currentOperator, currentOperatorFourier = simulation.CalculateTotalCurrent()

# ==============================================
# ==== PLOTTING EIGENBASIS CURRENT OPERATOR ====
# ==============================================

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

# plt.suptitle(title)
plt.tight_layout()
plt.show()

# Plotting the fourier transform of the current operator.
plt.semilogy(simulation.freqAxis / simulation.drivingFreq, np.abs(currentOperatorFourier)**2,
        color = 'black')

# plt.suptitle(title)
# plt.xlim(-2.5, 2.5)
plt.xlabel(r"$\omega / \Omega$")
plt.ylabel(r"$\| \tilde j (\omega) \|^2$")
plt.show()