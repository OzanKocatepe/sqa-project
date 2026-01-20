import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from typing import Callable

t1 = 0.2
t2 = 0.8 + 0j
drivingAmplitude = 1
drivingFreq = 0.5
k = np.pi / 4
decayConstant = 1

def ClassicalDrivingTerm(t: np.typing.ArrayLike) -> np.typing.ArrayLike:
    """
    The classical sinusoidal driving term that we are considering. This models a laser with frequency $\Omega$ and amplitude $A_0$.

    Parameters
    ----------
        t : ArrayLike
            The absolute value of time.

    Returns
    -------
        ArrayLike
            The value of the driving term at that time.
    """

    return drivingAmplitude * np.sin(drivingFreq * decayConstant * t)

def ClassicallyDrivenSSHEquations(t: float, c: np.ndarray[float], A: Callable[[np.typing.ArrayLike], np.typing.ArrayLike]) -> np.ndarray[float]:
    """
    The ODE (equations of motion) for the single-time expectations of $\sigma_-(t)$, $\sigma_+(t)$, and $\sigma_z(t)$. 

    Parameters
    ----------
        t : float
            The time t.
        c : ndarray[float, dtype[Any]]
            The 3-component single-time correlation array.
        A : Callable[[np.typing.ArrayLike], np.typing.ArrayLike]
            The classical driving term. Should be a vectorisable function of t.

    Returns:
    --------
        ndarraypfloat, dtype[Any]] 
            The value of dc/dt at some time t.
    """

    # Coefficient matrix at time t.
    Ek = t1 + t2 * np.exp(1j * k)
    phiK = np.angle(Ek)
    vZ = 2 * t2 * np.sin(k - phiK - 0.5 * A(t)) * np.sin(0.5 * A(t))

    B = np.array([[2j * (vZ - np.abs(Ek)) - 0.5 * decayConstant  , 0                                             ,  vZ                      ],
                  [0                                             , 2j * (np.abs(Ek) - vZ) - 0.5 * decayConstant  ,  vZ                      ],
                  [-2 * vZ                                       , -2 * vZ                                       ,  -decayConstant          ]], dtype=complex)
    
    # Inhomogenous part.
    d = np.array([0, 0, -decayConstant], dtype=complex)
    
    return B @ c + d

# ===============================
# ==== NUMERICALLY SOLVE ODE ====
# ===============================

# Define the choice of driving term.
# A = lambda t: 0
A = ClassicalDrivingTerm

# Here, the tDomain is defined in terms of $1 / \gamma_-$.
tDomain = np.array([0, 50])
n_tSamples = 250
tAxis = np.linspace(tDomain[0], tDomain[1], n_tSamples)
initialConditions = np.array([0, 0, -1], dtype=complex) # We assume that the system is in its ground state at time 0.

numericalSol = integrate.solve_ivp(fun=ClassicallyDrivenSSHEquations,
                                   t_span=tDomain / decayConstant,
                                   y0=initialConditions,
                                   t_eval=tAxis / decayConstant,
                                   rtol=1e-10,
                                   atol=1e-12,
                                   args=(A,))

# ==========================================
# ==== CALCULATING THE CURRENT OPERATOR ====
# ==========================================

# Calculates the current operator in terms of the pauli matrices in the eigenbasis.
currentCoeff = 1j * t2 * np.exp(1j * (k - A(tAxis / decayConstant)))
currentOperatorSol = -(currentCoeff * numericalSol.y[0] + currentCoeff.conjugate() * numericalSol.y[1])

# Takes the fourier transform of the current operator.
sampleSpacing = (tDomain[1] - tDomain[0]) * decayConstant / n_tSamples
fourierCurrentOperator = np.fft.fftshift(np.fft.fft(currentOperatorSol))
freqAxis = np.fft.fftshift(np.fft.fftfreq(n_tSamples, sampleSpacing))

# ===========================================
# ==== PLOTTING SINGLE-TIME CORRELATIONS ====
# ===========================================

# Writes the labels for each correlation that we are plotting.
correlationLabels = [r"$\langle \tilde \sigma_-(t) \rangle$",
                     r"$\langle \tilde \sigma_+(t) \rangle$",
                     r"$\langle \tilde \sigma_z(t) \rangle$"]

# Title of the plot.
title = rf"$t_1 = {t1},\, t_2 = {t2},\, A_0 = {drivingAmplitude},\, \Omega = {drivingFreq},\, k = {k / np.pi} \pi,\, \gamma_- = {decayConstant}$"

# Writes the labels for each individal subplot.
xLabel = r"$t / \gamma_-$"
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
    ax[row].plot(tAxis, plottingFunctions[row](currentOperatorSol),
                 color = "Black")
    
    ax[row].set_xlabel(xLabel)
    ax[row].set_ylabel(yLabels[row])

plt.suptitle(title)
plt.tight_layout()
plt.show()

# Plotting the fourier transform of the current operator.
plt.plot(freqAxis, np.abs(fourierCurrentOperator)**2,
        color = 'black')

plt.suptitle(title)
plt.xlim(-2.5, 2.5)
plt.xlabel("Frequency (Hz)")
plt.ylabel(r"$\| \tilde j (w) \|^2$")
plt.show()