import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpmath import invertlaplace

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
# The points (in the complex plane) that we evaluate the Laplaced solutions at.
sRange = 0.2
sNum = 50
sRealAxis, sImagAxis = np.meshgrid(np.linspace(-sRange, sRange, sNum), np.linspace(-sRange, sRange, 20))
sAxis = sRealAxis + 1j * sImagAxis

# Numerically solves the ODE for n = 1.
numericalSamples = integrate.solve_ivp(fun=TimeIndependentBlochEquations, t_span=tDomain, y0=initialConditions, t_eval=tAxis)

# =====================================
# ==== FINDING ANALYTICAL SOLUTION ====
# =====================================

# Defines the analytical solution for the Laplace of tilde sigma-minus as a function of s.
P = lambda s: (s + 2 / tau) * ( (s + 1 / tau)**2 + omegaTilde**2 ) + rabiFreq**2 * (s + 1 / tau)
analyticalLaplaceSol = lambda s: -0.5j * rabiFreq * (s + 2 / tau) * (s + 1 / tau - 1j * omegaTilde) / (s * P(s))
# Evaluates the analytical solution in the complex plane.
analyticalLaplaceSamples = analyticalLaplaceSol(sAxis)

# =================================================
# ==== LAPLACE TRANSFORM OF NUMERICAL SOLUTION ====
# =================================================

# Defines the laplace transformation of the function as a function of s.
numericalLaplaceSol = lambda s: integrate.simpson(
    numericalSamples.y[0] * np.exp(-s * numericalSamples.t), # Laplace transform integrand.
    numericalSamples.t                                       # Points at which numerical solution was evaluated.
)
# Evaluates the laplaced numerical solution at the desired points in the complex plane.
numericalLaplaceSamples = np.zeros(shape=(sAxis.shape), dtype=complex)
for x in np.arange(sAxis.shape[0]):
    for y in np.arange(sAxis.shape[1]):
        numericalLaplaceSamples[x, y] = numericalLaplaceSol(sAxis[x, y])

# ====================================================
# ==== INVERSE LAPLACE OF THE ANALYTICAL SOLUTION ====
# ====================================================

# Evaluates the inverse laplace transform of the equation in the papers for each time that we desire.
# Doesn't calculate at t = 0, because that ends up with a division by 0 apparently.
analyticalSamples = np.array([invertlaplace(analyticalLaplaceSol, t, method='talbot') for t in tAxis[1:]])

# ========================================
# ==== PLOTTING TIME DOMAIN SOLUTIONS ====
# ========================================

numColor = 'black'
analyticColor = 'blue'

magStyle = 'solid'
realStyle = 'dashed'
imagStyle = 'dotted'

# Creates the figure.
fig, ax = plt.subplots(3, 3, figsize=(16, 8.8))
# Rmoves the 2D axes and sets the 3D axes.
for x in [1, 2]:
    for y in [0, 1, 2]:
        ax[x, y].remove()
        ax[x, y] = fig.add_subplot(3, 3, 3 * x + y + 1, projection='3d')

tPlottingLimit = 30

# Magnitude of expectation.
ax[0, 0].plot(tAxis, np.abs(numericalSamples.y[0]), color=numColor, label='Numerical', linestyle=magStyle)
ax[0, 0].plot(tAxis[1:], np.abs(analyticalSamples), color=analyticColor, label='Analytical', linestyle=magStyle)
ax[0, 0].set_xlim(0, tPlottingLimit)

# R0, eal part of expectation.
ax[0, 1].plot(tAxis, numericalSamples.y[0].real, color=numColor, label='Numerical', linestyle=realStyle)
ax[0, 1].plot(tAxis[1:], analyticalSamples.real, color=analyticColor, label='Analytical', linestyle=realStyle)
ax[0, 1].set_xlim(0, tPlottingLimit)

# I0, maginary part of expectation.
ax[0, 2].plot(tAxis, numericalSamples.y[0].imag, color=numColor, label='Numerical', linestyle=imagStyle)
ax[0, 2].plot(tAxis[1:], analyticalSamples.imag, color=analyticColor, label='Analytical', linestyle=imagStyle)
ax[0, 2].set_xlim(0, tPlottingLimit)

xLabel = "$t / \\tau$"
ax[0, 0].set_xlabel(xLabel)
ax[0, 1].set_xlabel(xLabel)
ax[0, 2].set_xlabel(xLabel)

ax[0, 0].set_ylabel("$\\left\\| \\langle \\tilde{\\sigma}_-(t) \\rangle \\right\\|$")
ax[0, 1].set_ylabel("$\\text{Re} \\left[ \\langle \\tilde{\\sigma}_-(t) \\rangle \\right]$")
ax[0, 2].set_ylabel("$\\text{Im} \\left[ \\langle \\tilde{\\sigma}_-(t) \\rangle \\right]$")

ax[0, 0].legend()
ax[0, 1].legend()
ax[0, 2].legend()

# ===========================================
# ==== PLOTTING LAPLACE DOMAIN SOLUTIONS ====
# ===========================================

ax[1, 0].plot_surface(sAxis.real, sAxis.imag, np.abs(numericalLaplaceSamples), color=numColor, label="Numerical Magnitude")
ax[1, 0].plot_surface(sAxis.real, sAxis.imag, np.abs(analyticalLaplaceSamples), color=analyticColor, label="Analytical Magnitude")

ax[1, 1].plot_surface(sAxis.real, sAxis.imag, numericalLaplaceSamples.real, color=numColor, label="Numerical Real Part")
ax[1, 1].plot_surface(sAxis.real, sAxis.imag, analyticalLaplaceSamples.real, color=analyticColor, label="Analytical Real Part")

ax[1, 2].plot_surface(sAxis.real, sAxis.imag, numericalLaplaceSamples.imag, color=numColor, label="Numerical Imaginary Part")
ax[1, 2].plot_surface(sAxis.real, sAxis.imag, analyticalLaplaceSamples.imag, color=analyticColor, label="Analytical Imaginary Part")

xLabel = "$\\text{Re}(s)$"
ax[1, 0].set_xlabel(xLabel)
ax[1, 1].set_xlabel(xLabel)
ax[1, 2].set_xlabel(xLabel)

yLabel = "$\\text{Im}(s)$"
ax[1, 0].set_ylabel(yLabel)
ax[1, 1].set_ylabel(yLabel)
ax[1, 2].set_ylabel(yLabel)

ax[1, 0].set_zlabel("$\\left\\| \\mathcal{L}_t \\langle \\tilde{\\sigma}_-(t) \\rangle \\right\\|$")
ax[1, 1].set_zlabel("$\\text{Re} \\left[ \\mathcal{L}_t \\langle \\tilde{\\sigma}_-(t) \\rangle \\right]$")
ax[1, 2].set_zlabel("$\\text{Im} \\left[ \\mathcal{L}_t \\langle \\tilde{\\sigma}_-(t) \\rangle \\right]$")

ax[1, 0].legend()
ax[1, 1].legend()
ax[1, 2].legend()

# ============================================
# ==== PLOTTING LAPLACE DOMAIN DIFFERENCE ====
# ============================================

zeroColor = 'red'
key = np.abs(numericalLaplaceSamples - analyticalLaplaceSamples) <= 1e-3

# Maps the key values to colors.
# colorMap = np.zeros((sAxis.shape[0], sAxis.shape[1], 3), dtype=float)
# colorMap[key] = mcolors.to_rgb(zeroColor)
# colorMap[~key] = mcolors.to_rgb(numColor)

# Defines the upper and lower range for which we would like to see the range of differences.
range = (-1, 1)

# Plots the differences.
ax[2, 0].plot_surface(sAxis.real, sAxis.imag, np.abs(numericalLaplaceSamples - analyticalLaplaceSamples), cmap='viridis', vmin=range[0], vmax=range[1], shade=False)
ax[2, 1].plot_surface(sAxis.real, sAxis.imag, (numericalLaplaceSamples - analyticalLaplaceSamples).real, cmap='viridis', vmin=range[0], vmax=range[1], shade=False)
ax[2, 2].plot_surface(sAxis.real, sAxis.imag, (numericalLaplaceSamples - analyticalLaplaceSamples).imag, cmap='viridis', vmin=range[0], vmax=range[1], shade=False)

# Limits the z-axis.
ax[2, 0].set_zlim(range)
ax[2, 1].set_zlim(range)
ax[2, 2].set_zlim(range)

xLabel = "$\\text{Re}(s)$"
ax[2, 0].set_xlabel(xLabel)
ax[2, 1].set_xlabel(xLabel)
ax[2, 2].set_xlabel(xLabel)

yLabel = "$\\text{Im}(s)$"
ax[2, 0].set_ylabel(yLabel)
ax[2, 1].set_ylabel(yLabel)
ax[2, 2].set_ylabel(yLabel)

ax[2, 0].set_zlabel("Norm of Difference")
ax[2, 1].set_zlabel("Real Part of Difference")
ax[2, 2].set_zlabel("Imaginary Part of Difference")

plt.tight_layout()
plt.show()

# ===========================================
# ==== PLOTTING HISTOGRAM OF DIFFERENCES ====
# ===========================================

fig, ax = plt.subplots(3, 1, figsize=(16, 8.8))

bins = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
ax[0].hist(np.abs(numericalLaplaceSamples - analyticalLaplaceSamples).flatten(), bins=bins, color='blue')
ax[0].set_xlabel("Norm of Difference")
ax[0].set_ylabel("Frequency")
ax[0].set_xscale('log')

ax[1].hist((numericalLaplaceSamples - analyticalLaplaceSamples).real.flatten(), bins=bins, color='green')
ax[1].set_xlabel("Real Part of Difference")
ax[1].set_ylabel("Frequency")
ax[1].set_xscale('log')

ax[2].hist((numericalLaplaceSamples - analyticalLaplaceSamples).imag.flatten(), bins=bins, color='red')
ax[2].set_xlabel("Imaginary Part of Difference")
ax[2].set_ylabel("Frequency")
ax[2].set_xscale('log')

plt.tight_layout()
plt.show()