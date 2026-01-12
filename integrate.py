import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpmath import invertlaplace
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sympy

tau = 1        # Characteristic decay scale.
omegaTilde = 0 # Atomic transition frequency - photon frequency.
rabiFreq = 2

def TimeIndependentBlochEquations(t: np.typing.ArrayLike, c: np.ndarray[float]) -> np.ndarray[float]:
    """
    Function for the optical Bloch equations with radiative damping from
    the (Kocabas et. al., 2012) paper - https://link.aps.org/doi/10.1103/PhysRevA.85.023817.
    These are the time-independent form of the equations from Appendix C.

    Parameters
    ----------
        t : ArrayLike
            The time (t') that we are evaluating the ODE at. Since the equations are time-independent,
            this should only impact the inhomogenous term.
        c : ndarray[float, dtype[Any]]
            The array of correlations.

    Returns
    -------
        ndarray[float, dtype[Any]]
            The value of dc/dt at the given time t.
    """

    # Ensures t is a numpy array.
    t = np.array(t)

    # Common term defined for convenience.
    omegaTerm = -(1 / tau + 1j * omegaTilde)

    # Coefficient matrix (NOT IN PARAMETERISED FORM - CHANGE ONCE SOLUTIONS MATCH FOR TAU = 1)
    M = np.array([[omegaTerm    ,    0                    ,    0.5j * rabiFreq ],
                  [0            ,    omegaTerm.conjugate(),    -0.5j * rabiFreq],
                  [1j * rabiFreq,    -1j * rabiFreq       ,    -2 / tau        ]], dtype=complex)

    # Inhomogenous coefficient.
    b = np.array([0, 0, - 2 / tau])

    return M @ c + b

# ====================================
# ==== FINDING NUMERICAL SOLUTION ====
# ====================================

# Atom is in ground state at t = 0.
initialConditions = np.array([0, 0, -1], dtype=complex)

# The points within the range (0, timeLimit) we will evaluate the numerical solution at.
tDomain = (0, 300)
tAxis = np.linspace(tDomain[0], tDomain[1], 1000)

# Numerically solves the ODE for n = 1.
numericalSamples = integrate.solve_ivp(fun=TimeIndependentBlochEquations,
                                       t_span=tDomain,
                                       y0=initialConditions,
                                       t_eval=tAxis,
                                       rtol=1e-10,
                                       atol=1e-12)

# =========================================
# ==== FINDING THE ANALYTICAL SOLUTION ====
# =========================================

# Defines the analytical solution symbolically using sympy.
s, tSym, tauSym, omegaTildeSym, rabiFreqSym = sympy.symbols('s, t, tau, omegaTilde, rabiFreq')
PSym = (s + 2 / tauSym) * ( (s + 1 / tauSym)**2 + omegaTildeSym**2) + rabiFreqSym**2 * (s + 1 / tauSym)
analyticalLaplaceSolSym = -0.5j * rabiFreqSym * (s + 2 / tauSym) * (s + 1 / tauSym - 1j * omegaTildeSym) / (s * PSym)

# Simplifies the expression.
analyticalLaplaceSolSym = sympy.simplify(analyticalLaplaceSolSym)

# Substitutes the numerical values of our parameters.
analyticalLaplaceSolSym = analyticalLaplaceSolSym.subs([
    (tauSym, tau),
    (omegaTildeSym, omegaTilde),
    (rabiFreqSym, rabiFreq)
])

# Performs partial fraction decomposition.
analyticalLaplaceSolSym = sympy.apart(analyticalLaplaceSolSym, s)

# Simplifies with the new numerical values.
analyticalLaplaceSolSym = sympy.simplify(analyticalLaplaceSolSym)

# Takes the inverse laplace transform symbolically.
analyticalSolSym = sympy.inverse_laplace_transform(analyticalLaplaceSolSym, s, tSym)

# Converts it to a numerical function.
analyticalFunc = sympy.lambdify(tSym, analyticalSolSym, modules=['numpy'])
# Calculates the desired time domain values.
analyticalSamples = analyticalFunc(tAxis)

# ========================================
# ==== PLOTTING TIME DOMAIN SOLUTIONS ====
# ========================================

numericalColor = 'black'
analyticalColor = 'blue'

figSize = (16, 8.8)
tPlottingRanges = [5, 30, 100] # The range each row should plot.
plottingFunctions = [np.abs, lambda z: z.real, lambda z: z.imag] # The function each column should plot.
tLineStyles = ['solid', 'dashed', 'dotted'] # The line style for each function/column.

xLabel = r"$t / \tau$" # x-axis label.
yLabels = [r"$\left| \langle \tilde{\sigma_-(t)} \rangle \right|$",
           r"$\text{Re} \left[ \langle \tilde{\sigma_-(t)} \rangle \right]$",
           r"$\text{Im} \left[ \langle \tilde{\sigma_-(t)} \rangle \right]$"]

# Creates the figure.
fig, ax = plt.subplots(len(tPlottingRanges), len(plottingFunctions), figsize=figSize)

# Looping through the subplots.
for row in np.arange(len(tPlottingRanges)):
    for col in np.arange(len(plottingFunctions)):
        # Plot numerical solution.
        ax[row, col].plot(tAxis, plottingFunctions[col](numericalSamples.y[0]),
                          color = numericalColor,
                          label = "Numerical",
                          linestyle = tLineStyles[col])
        
        # Plots analytical solution.
        ax[row, col].plot(tAxis, plottingFunctions[col](analyticalSamples),
                          color = analyticalColor,
                          label = "Analytical",
                          linestyle = tLineStyles[col])
        
        # Sets other properties.
        ax[row, col].set_xlim(0, tPlottingRanges[row])
        ax[row, col].set_xlabel(xLabel)
        ax[row, col].set_ylabel(yLabels[col])
        ax[row, col].legend()

plt.tight_layout()
plt.show()