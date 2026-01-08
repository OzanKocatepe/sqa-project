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
tAxis = np.linspace(0, tDomain[1], 1000)
# The points (in the complex plane) that we evaluate the Laplaced solutions at.
sRange = 0.2
sRealAxis, sImagAxis = np.meshgrid(np.linspace(-sRange, sRange, 20), np.linspace(-sRange, sRange, 20))
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
analyticalSamples = np.array([invertlaplace(analyticalLaplaceSol, t) for t in tAxis[1:]])

# ========================================
# ==== PLOTTING TIME DOMAIN SOLUTIONS ====
# ========================================

numColor = 'black'
analyticColor = 'blue'

magStyle = 'solid'
realStyle = 'dashed'
imagStyle = 'dotted'

fig, ax = plt.subplots(1, 3, figsize=(32, 6))

# Magnitude of expectation.
ax[0].plot(tAxis, np.abs(numericalSamples.y[0]), color=numColor, label='Numerical Magnitude', linestyle=magStyle)
ax[0].plot(tAxis[1:], np.abs(analyticalSamples), color=analyticColor, label='Analytical Magnitude', linestyle=magStyle)

# Real part of expectation.
ax[1].plot(tAxis, numericalSamples.y[0].real, color=numColor, label='Numerical Real Part', linestyle=realStyle)
ax[1].plot(tAxis[1:], analyticalSamples.real, color=analyticColor, label='Analytical Real Part', linestyle=realStyle)

# Imaginary part of expectation.
ax[2].plot(tAxis, numericalSamples.y[0].imag, color=numColor, label='Numerical Imaginary Part', linestyle=imagStyle)
ax[2].plot(tAxis[1:], analyticalSamples.imag, color=analyticColor, label='Analytical Imaginary Part', linestyle=imagStyle)

xLabel = "$t / \\tau$"
ax[0].set_xlabel(xLabel)
ax[1].set_xlabel(xLabel)
ax[2].set_xlabel(xLabel)

ax[0].set_ylabel("$\\left\\| \\langle \\tilde{\\sigma}_-(t) \\rangle \\right\\|$")
ax[1].set_ylabel("\\text{Re} \\left[ \\langle \\tilde{\\sigma}_-(t) \\rangle \\right]")
ax[2].set_ylabel("\\text{Im} \\left[ \\langle \\tilde{\\sigma}_-(t) \\rangle \\right]")

ax[0].legend()
ax[1].legend()
ax[2].legend()

plt.show()

# ===========================================
# ==== PLOTTING LAPLACE DOMAIN SOLUTIONS ====
# ===========================================

fig, ax = plt.subplots(1, 3, figsize=(32, 6), subplot_kw={'projection' : '3d'})

ax[0].plot_surface(sAxis.real, sAxis.imag, np.abs(numericalLaplaceSamples), color=numColor, label="Numerical Magnitude")
ax[0].plot_surface(sAxis.real, sAxis.imag, np.abs(analyticalLaplaceSamples), color=analyticColor, label="Analytical Magnitude")

ax[1].plot_surface(sAxis.real, sAxis.imag, numericalLaplaceSamples.real, color=numColor, label="Numerical Magnitude")
ax[1].plot_surface(sAxis.real, sAxis.imag, analyticalLaplaceSamples.real, color=analyticColor, label="Analytical Magnitude")

ax[2].plot_surface(sAxis.real, sAxis.imag, numericalLaplaceSamples.imag, color=numColor, label="Numerical Magnitude")
ax[2].plot_surface(sAxis.real, sAxis.imag, analyticalLaplaceSamples.imag, color=analyticColor, label="Analytical Magnitude")

xLabel = "$\\text{Re}(s)$"
ax[0].set_xlabel(xLabel)
ax[1].set_xlabel(xLabel)
ax[2].set_xlabel(xLabel)

yLabel = "$\\text{Im}(s)$"
ax[0].set_ylabel(yLabel)
ax[1].set_ylabel(yLabel)
ax[2].set_ylabel(yLabel)

ax[0].set_zlabel("$\\left\\| \\mathcal{L}_t \\langle \\tilde{\\sigma}_-(t) \\rangle \\right\\|$")
ax[1].set_zlabel("$\\text{Re} \\left[ \\mathcal{L}_t \\langle \\tilde{\\sigma}_-(t) \\rangle \\right]$")
ax[2].set_zlabel("$\\text{Im} \\left[ \\mathcal{L}_t \\langle \\tilde{\\sigma}_-(t) \\rangle \\right]$")

ax[0].legend()
ax[1].legend()
ax[2].legend()

plt.show()

# ============================================
# ==== PLOTTING LAPLACE DOMAIN DIFFERENCE ====
# ============================================

zeroColor = 'red'
fig, ax = plt.subplots(1, 3, figsize=(32, 6), subplot_kw={'projection' : '3d'})
key = np.abs(numericalLaplaceSamples - analyticalLaplaceSamples) <= 1e-3

# Maps the key values to colors.
colorMap = np.zeros((sAxis.shape[0], sAxis.shape[1], 3), dtype=float)
colorMap[key] = mcolors.to_rgb(zeroColor)
colorMap[~key] = mcolors.to_rgb(numColor)

ax[0].plot_surface(sAxis.real, sAxis.imag, np.abs(numericalLaplaceSamples - analyticalLaplaceSamples), facecolors=colorMap, shade=False)
ax[1].plot_surface(sAxis.real, sAxis.imag, (numericalLaplaceSamples - analyticalLaplaceSamples).real, facecolors=colorMap, shade=False)
ax[2].plot_surface(sAxis.real, sAxis.imag, (numericalLaplaceSamples - analyticalLaplaceSamples).imag, facecolors=colorMap, shade=False)

xLabel = "$\\text{Re}(s)$"
ax[0].set_xlabel(xLabel)
ax[1].set_xlabel(xLabel)
ax[2].set_xlabel(xLabel)

yLabel = "$\\text{Im}(s)$"
ax[0].set_ylabel(yLabel)
ax[1].set_ylabel(yLabel)
ax[2].set_ylabel(yLabel)

ax[0].set_zlabel("Norm of Difference")
ax[1].set_zlabel("Real Part of Difference")
ax[2].set_zlabel("Imaginary Part of Difference")

plt.show()