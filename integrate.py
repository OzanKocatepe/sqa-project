import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpmath import invertlaplace
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
sRange = 5
sNum = 200
sRealAxis, sImagAxis = np.meshgrid(np.linspace(-sRange, sRange, sNum), np.linspace(-sRange, sRange, sNum))
sAxis = sRealAxis + 1j * sImagAxis

# Numerically solves the ODE for n = 1.
numericalSamples = integrate.solve_ivp(fun=TimeIndependentBlochEquations,
                                       t_span=tDomain,
                                       y0=initialConditions,
                                       t_eval=tAxis,
                                       rtol=1e-10,
                                       atol=1e-12)

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
analyticalSamples = np.array([complex(invertlaplace(analyticalLaplaceSol, float(t), method='talbot')) 
                              for t in tAxis[1:]])

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
        ax[row, col].plot(tAxis[1:], plottingFunctions[col](analyticalSamples),
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

# ===========================================
# ==== PLOTTING LAPLACE DOMAIN SOLUTIONS ====
# ===========================================

zLabels = ["Norm of Sigma",
           "Re(Sigma)",
           "Im(Sigma)"]

for page in np.arange(len(plottingFunctions)):
    # Plots the analytical and numerical solutions separately on the same page.
    fig = make_subplots(
        rows = 1, cols = 2,
        specs = [[{'type' : 'surface'}, {'type' : 'surface'}]],
        subplot_titles = ("Analytical Solution", "Numerical Solution")
    )

    # Plotting the analytical solution.
    fig.add_trace(
        go.Surface(z = plottingFunctions[page](analyticalLaplaceSamples), x = sAxis.real, y = sAxis.imag, showscale = False),
        row = 1, col = 1
    )

    # Plotting the numerical solution.
    # mask = np.abs(numericalLaplaceSamples) >= 0
    fig.add_trace(
        go.Surface(z = plottingFunctions[page](numericalLaplaceSamples), x = sAxis.real, y = sAxis.imag, showscale = False),
        row = 1, col = 2
    )

    # Sets the layout of the plot.
    fig.update_layout(
        width=1500, height=800,
        scene = dict(
            xaxis = dict( title = "Re(s)" ),
            yaxis = dict( title = "Im(s)" ),
            zaxis = dict( title = zLabels[page])
        ),
        scene2 = dict(
            xaxis = dict( title = "Re(s)" ),
            yaxis = dict( title = "Im(s)" ),
            zaxis = dict( title = zLabels[page])
        )
    )
    
    fig.show()

quit()

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