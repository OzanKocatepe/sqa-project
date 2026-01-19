import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from typing import Callable

t1 = 0.2
t2 = 0.8 + 0j
drivingAmplitude = 5
drivingFreq = 0.5
k = np.pi / 4

def ClassicalDrivingTerm(t: np.typing.ArrayLike) -> np.typing.ArrayLike:
    """
    The classical sinusoidal driving term that we are considering. This models a laser with frequency $\Omega$ and amplitude $A_0$.

    Parameters
    ----------
        t : ArrayLike
            The time.

    Returns
    -------
        ArrayLike
            The value of the driving term at that time.
    """

    return drivingAmplitude * np.sin(drivingFreq * t)

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
    vPm = 2j * t2 * np.cos(k - phiK - 0.5 * A(t)) * np.sin(0.5 * A(t))

    B = np.array([[2j * (vZ - np.abs(Ek))  , 0                       ,  -1j * vPm  ],
                  [0                       , 2j * (np.abs(Ek) - vZ)  ,  -1j * vPm  ],
                  [2j * vPm                , 2j * vPm                ,  0          ]], dtype=complex)
    
    return B @ c

# ===============================
# ==== NUMERICALLY SOLVE ODE ====
# ===============================

# Define the choice of driving term.
A = ClassicalDrivingTerm
tDomain = (0, 5)
tAxis = np.linspace(tDomain[0], tDomain[1], 250)
initialConditions = np.array([0, 0, -1], dtype=complex) # We assume that the system is in its ground state at time 0.

numericalSol = integrate.solve_ivp(fun=ClassicallyDrivenSSHEquations,
                                    t_span=tDomain,
                                    y0=initialConditions,
                                    t_eval=tAxis,
                                    rtol=1e-10,
                                    atol=1e-12,
                                    args=(A,))

# ==========================================
# ==== CALCULATING THE CURRENT OPERATOR ====
# ==========================================

# Calculates the current operator in terms of the pauli matrices in the eigenbasis.
currentCoeff = 1j * t2 * np.exp(1j * (k - A(tAxis)))
currentOperatorSol = -(currentCoeff * numericalSol.y[0] + currentCoeff.conjugate() * numericalSol.y[1])

# ===========================================
# ==== PLOTTING SINGLE-TIME CORRELATIONS ====
# ===========================================

# Writes the labels for each correlation that we are plotting.
correlationLabels = [r"$\langle \tilde \sigma_-(t) \rangle$",
                     r"$\langle \tilde \sigma_+(t) \rangle$",
                     r"$\langle \tilde \sigma_z(t) \rangle$"]

# Writes the labels for each individal subplot.
xLabel = "$t$"
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

plt.tight_layout()
plt.show()

# ===================================
# ==== PLOTTING CURRENT OPERATOR ====
# ===================================

currentLabel = r"$\langle \tilde j_k \rangle$"
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

plt.tight_layout()
plt.show()