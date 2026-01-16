import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import sympy

tau = 2             # Characteristic decay scale.
detuningFreq = 1    # Normalised detuning frequency, described in Appendix B. Denoted D in the paper.
rabiFreq = 5        # Normalised rabi frequency, described in appendix B. Denoted R in the paper.

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

"""
Loop through the single and double time correlations.
The most important distinctions between these are:
    - The single time correlation is a function of t', while the double time correlations are functions of t and t'.
      However, we choose t -> inf and t' = t + dt, so in a sense both are just a function of dt.
    - Single and double time correlations have different initial conditions
      and inhomogenous terms.
"""
for n in np.arange(2):
    if n == 0:
        # Atom is in ground state at t = 0.
        initialConditions = np.array([0, 0, -1], dtype=complex)

        # The inhomogenous term is just a constant.
        inhomTerm = -2 / tau
    elif n == 1:
        # We take the initial condition as t' = t, with t -> inf, and so
        # we can find the initial conditions using the steady-state solutions
        # for the single time correlations, with some algebra.
        initialConditions = np.array([0.25 * rabiFreq**2 / (1 + detuningFreq**2 + 0.5 * rabiFreq**2),
                                      0,
                                      -0.5j * rabiFreq * (1 + 1j * detuningFreq) / (1 + detuningFreq**2 + 0.5 * rabiFreq**2)], dtype=complex)
        
        # The inhomogenous term is sigma_+ at time t. Since for our initial conditions we are considering t -> infinity, we are
        # always considering a t large enough that we can use the steady-state form of sigma_+, so we have a closed form for
        # the inhomogenous term.
        inhomTerm = -1 / tau * 1j * rabiFreq * (1 + 1j * detuningFreq) / (1 + detuningFreq**2 + 0.5 * rabiFreq**2)
        
    """
    For the single time correlations (n = 1), we treat t = 0 and plot t' = 0 + dt, so we
    simply label the time axis t. For the double time correlations (n = 2), we treat the initial time
    as some t and take the steady-state solution as t -> inf, and then plot as a function of dt (i.e. how
    separated the two operators are in time). For both scenarios, we are plotting the correlations
    as a function of the time separation, dt).
    """

    # The points within the range (0, timeLimit) we will evaluate the numerical solution at.
    tDomain = np.array([1e-6, 10])
    tAxis = np.linspace(tDomain[0], tDomain[1], 50)

    # Numerically solves the ODE.
    numericalSol = integrate.solve_ivp(fun=TimeIndependentBlochEquations,
                                       t_span=tDomain * tau,
                                       y0=initialConditions,
                                       t_eval=tAxis * tau,
                                       rtol=1e-10,
                                       atol=1e-12,
                                       args=(inhomTerm,))

    # =========================================
    # ==== FINDING THE ANALYTICAL SOLUTION ====
    # =========================================

    # Defines the analytical solutions symbolically using sympy.
    s, tSym, tauSym, detuningFreqSym, rabiFreqSym = sympy.symbols('s, t, tau, detuningFreq, rabiFreq')

    # P is the function P(s) defined in equation (B4).
    P = (s + 2 / tauSym) * ( (s + 1 / tauSym)**2 + detuningFreqSym**2 / tauSym**2) + rabiFreqSym**2 / tauSym**2 * (s + 1 / tauSym)

    if n == 0:
        # This is a vector containing the equations (B1-B3), which are the analytical solutions for the
        # single-time correlation functions.
        analyticalLaplaceSym = sympy.Matrix([-0.5j * rabiFreqSym / tauSym * (s + 2 / tauSym) * (s + 1 / tauSym - 1j * detuningFreqSym / tauSym) / (s * P),
                                            0.5j * rabiFreqSym / tauSym * (s + 2 / tauSym) * (s + 1 / tauSym + 1j * detuningFreqSym / tauSym) / (s * P),
                                            -(s + 2 / tauSym) * ( (s + 1 / tauSym)**2 + detuningFreqSym**2 / tauSym**2) / (s * P)])
    elif n == 1:
        # This is a 'vector' containing just the analytical solution for the first component of the
        # double-time correlation functions. We have chosen at this time not to plot the remaining analytical
        # solutions.
        analyticalLaplaceSym = sympy.Matrix([ 0.25 * rabiFreqSym**2 * (P - 0.5 * rabiFreqSym**2 * (s + 2 / tauSym) / tauSym**2) / (s * P * (1 + detuningFreqSym**2 + 0.5 * rabiFreqSym**2)) ])

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

    # Converts it to a numerical function. numpy module would apparently be faster, but the way it deals with
    # complex values breaks this line when D != 0.
    analyticalFunc = sympy.lambdify(tSym, analyticalSolSym, modules=['sympy'])
    # Creates an array to store the samples in, of the right shape for a sympy matrix output.
    analyticalSamples = np.zeros((3, 1, tAxis.size), dtype=complex)
    # Calculates each t value manually, because for some reason putting in a vector,
    # at least while the module is sympy, breaks the line.
    for i in np.arange(tAxis.size):
        if (round(i / tAxis.size * 100, 1) == round(i / tAxis.size * 100)):
            print(f"{i / tAxis.size * 100:.0f}%")
        analyticalSamples[:, :, i] = analyticalFunc(tAxis[i] * tau)

    # ========================================
    # ==== PLOTTING TIME DOMAIN SOLUTIONS ====
    # ========================================

    numericalColor = 'black'
    analyticalColor = 'blue'

    figSize = (16, 8.8)
    plottingFunctions = [np.abs, lambda z: z.real, lambda z: z.imag] # The function each column should plot.
    tLineStyles = ['solid', 'dashed', 'dotted'] # The line style for each function/column.
    titles = ["Single-Time Correlation Functions", "Double-Time Correlation Functions"]
    xLabel = r"$\delta t / \tau$" # x-axis label.

    if n == 0:
        yLabels = [[r"$\left\| \langle \tilde{\sigma}_-(\delta t) \rangle \right\|$", r"$\text{Re} \left[ \langle \tilde{\sigma}_-(\delta t) \rangle \right]$", r"$\text{Im} \left[ \langle \tilde{\sigma}_-(\delta t) \rangle \right]$"],
                   [r"$\left\| \langle \tilde{\sigma}_+(\delta t) \rangle \right\|$", r"$\text{Re} \left[ \langle \tilde{\sigma}_+(\delta t) \rangle \right]$", r"$\text{Im} \left[ \langle \tilde{\sigma}_+(\delta t) \rangle \right]$"],
                   [r"$\left\| \langle \tilde{\sigma}_z(\delta t) \rangle \right\|$", r"$\text{Re} \left[ \langle \tilde{\sigma}_z(\delta t) \rangle \right]$", r"$\text{Im} \left[ \langle \tilde{\sigma}_z(\delta t) \rangle \right]$"]]
    elif n == 1:
        yLabels = [[r"$\left\| \langle \tilde{\sigma}_+(t) \tilde{\sigma}_-(t + \delta t) \rangle \right\|$", r"$\text{Re} \left[ \langle \tilde{\sigma}_+(t) \tilde{\sigma}_-(t + \delta t) \rangle \right]$", r"$\text{Im} \left[ \langle \tilde{\sigma}_+(t) \tilde{\sigma}_-(t + \delta t) \rangle \right]$"],
                   [r"$\left\| \langle \tilde{\sigma}_+(t) \tilde{\sigma}_+(t + \delta t) \rangle \right\|$", r"$\text{Re} \left[ \langle \tilde{\sigma}_+(t) \tilde{\sigma}_+(t + \delta t) \rangle \right]$", r"$\text{Im} \left[ \langle \tilde{\sigma}_+(t) \tilde{\sigma}_+(t + \delta t) \rangle \right]$"],
                   [r"$\left\| \langle \tilde{\sigma}_+(t) \tilde{\sigma}_z(t + \delta t) \rangle \right\|$", r"$\text{Re} \left[ \langle \tilde{\sigma}_+(t) \tilde{\sigma}_z(t + \delta t) \rangle \right]$", r"$\text{Im} \left[ \langle \tilde{\sigma}_+(t) \tilde{\sigma}_z(t + \delta t) \rangle \right]$"]]

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
        
            # Plots analytical solution, as long as it is not the second or third components of
            # the double-time correlation vector, since we have chosen not to compare those
            # to analytical solutions.
            if not (n == 1 and (row == 1 or row == 2)):
                ax[row, col].plot(tAxis, plottingFunctions[col](analyticalSamples[row, 0, :]),
                                color = analyticalColor,
                                label = "Analytical Sol",
                                linestyle = tLineStyles[col])
        
            # Sets other properties.
            ax[row, col].set_xlabel(xLabel)
            ax[row, col].set_ylabel(yLabels[row][col])
            ax[row, col].legend()

    plt.suptitle(fr"$\tau = {tau}$, $D = {detuningFreq}$, $R = {rabiFreq}$")
    plt.tight_layout()
    plt.savefig(titles[n], dpi=300)
    plt.show()