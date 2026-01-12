import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import sympy

tau = 1             # Characteristic decay scale.
detuningFreq = 0    # Normalised detuning frequency, described in Appendix B.
rabiFreq = 2        # Normalised rabi frequency, described in appendix B.

def TimeIndependentBlochEquations(t: np.typing.ArrayLike, c: np.ndarray[float], b: float) -> np.ndarray[float]:
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
        b : float
            The inhomogenous term. The inhomogenous term vector is (0, 0, b)

    Returns
    -------
        ndarray[float, dtype[Any]]
            The value of dc/dt at the given time t.
    """

    # Ensures t is a numpy array.
    t = np.array(t)

    # Common term defined for convenience.
    omegaTerm = -(1 + 1j * detuningFreq)

    # Coefficient matrix (in form parameterised by tau).
    M = 1 / tau * np.array([[omegaTerm      , 0                      , 0.5j * rabiFreq  ],
                            [0              , omegaTerm.conjugate()  , -0.5j * rabiFreq ],
                            [1j * rabiFreq  , -1j * rabiFreq         , -2               ]], dtype=complex)

    # Inhomogenous coefficient.
    inhomTerm = np.array([0, 0, b])

    return M @ c + inhomTerm

# ====================================
# ==== FINDING NUMERICAL SOLUTION ====
# ====================================

# Atom is in ground state at t = 0.
initialConditions = np.array([0, 0, -1], dtype=complex)

# The points within the range (0, timeLimit) we will evaluate the numerical solution at.
tDomain = (0, 50)
tAxis = np.linspace(tDomain[0], tDomain[1], 1000)

# Numerically solves the ODE for n = 1.
numericalSol = integrate.solve_ivp(fun=TimeIndependentBlochEquations,
                                   t_span=tDomain,
                                   y0=initialConditions,
                                   t_eval=tAxis,
                                   rtol=1e-10,
                                   atol=1e-12,
                                   args=(-2 / tau,))

# =========================================
# ==== FINDING THE ANALYTICAL SOLUTION ====
# =========================================

# Defines the analytical solutions symbolically using sympy.
s, tSym, tauSym, detuningFreqSym, rabiFreqSym = sympy.symbols('s, t, tau, detuningFreq, rabiFreq')

# P is the function P(s) defined in equation (B4)
P = (s + 2 / tauSym) * ( (s + 1 / tauSym)**2 + detuningFreqSym**2) + rabiFreqSym**2 * (s + 1 / tauSym)
# This is a vector containing the equations (B1-B3)
analyticalLaplaceSym = sympy.Matrix([-0.5j * rabiFreqSym * (s + 2 / tauSym) * (s + 1 / tauSym - 1j * detuningFreqSym) / (s * P),
                                     0.5j * rabiFreqSym * (s + 2 / tauSym) * (s + 1 / tauSym + 1j * detuningFreqSym) / (s * P),
                                     -(s + 2 / tauSym) * ( (s + 1 / tauSym)**2 + detuningFreq**2) / (s * P)])

# Simplifies the expression.
analyticalLaplaceSym = sympy.simplify(analyticalLaplaceSym)

# Substitutes the numerical values of our parameters.
analyticalLaplaceSym = analyticalLaplaceSym.subs([
    (tauSym, tau),
    (detuningFreqSym, detuningFreq),
    (rabiFreqSym, rabiFreq)
])

# Performs partial fraction decomposition.
analyticalLaplaceSym = sympy.apart(analyticalLaplaceSym, s)

# Simplifies with the new numerical values.
analyticalLaplaceSym = sympy.simplify(analyticalLaplaceSym)

# Takes the inverse laplace transform symbolically.
analyticalSolSym = sympy.inverse_laplace_transform(analyticalLaplaceSym, s, tSym)

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
tLimit = 10
plottingFunctions = [np.abs, lambda z: z.real, lambda z: z.imag] # The function each column should plot.
tLineStyles = ['solid', 'dashed', 'dotted'] # The line style for each function/column.

xLabel = r"$t / \tau$" # x-axis label.
yLabels = [[r"$\left\| \langle \tilde{\sigma}_-(t) \rangle \right\|$", r"$\text{Re} \left[ \langle \tilde{\sigma}_-(t) \rangle \right]$", r"$\text{Im} \left[ \langle \tilde{\sigma}_-(t) \rangle \right]$"],
           [r"$\left\| \langle \tilde{\sigma}_+(t) \rangle \right\|$", r"$\text{Re} \left[ \langle \tilde{\sigma}_+(t) \rangle \right]$", r"$\text{Im} \left[ \langle \tilde{\sigma}_+(t) \rangle \right]$"],
           [r"$\left\| \langle \tilde{\sigma}_z(t) \rangle \right\|$", r"$\text{Re} \left[ \langle \tilde{\sigma}_z(t) \rangle \right]$", r"$\text{Im} \left[ \langle \tilde{\sigma}_z(t) \rangle \right]$"]]

# Creates the figure.
nrows = 3
ncols = len(plottingFunctions)
fig, ax = plt.subplots(nrows, ncols, figsize=figSize)

# Looping through the subplots.
for row in np.arange(nrows):
    for col in np.arange(ncols):
        # Plot numerical solution.
        ax[row, col].plot(tAxis, plottingFunctions[col](numericalSol.y[row]),
                          color = numericalColor,
                          label = "Numerical Sol",
                          linestyle = tLineStyles[col])
        
        # Plots analytical solution.
        ax[row, col].plot(tAxis, plottingFunctions[col](analyticalSamples[row, 0, :]),
                          color = analyticalColor,
                          label = "Analytical Sol",
                          linestyle = tLineStyles[col])
        
        # Sets other properties.
        ax[row, col].set_xlim(0, tLimit)
        ax[row, col].set_xlabel(xLabel)
        ax[row, col].set_ylabel(yLabels[row][col])
        ax[row, col].legend()

plt.suptitle(fr"$\tau = {tau}$, $D = {detuningFreq}$, $R = {rabiFreq}$")
plt.tight_layout()
# plt.savefig("Single-Time Correlation Functions", dpi=300)
plt.show()