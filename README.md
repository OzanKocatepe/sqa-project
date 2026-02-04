# SQA Project
This repo contains the code I used to model resonance fluorescence in a variety of systems. Currently, we can model a qubit in a waveguide (following [Kocabas et. al., 2012](https://arxiv.org/abs/1111.7315))
and are currently in the process of modelling a one-dimensional Su–Schrieffer–Heeger (SSH) model with an infinite bulk.

# Progress

Currently searching for the error in the code. Potential problems include:

- [x] The function which manually calculates the fourier transforms of the connected current correlator at the integer harmonics,
could be functioning incorrectly.

  - Tested the function in HarmonicTest.py, function works fine for normal signals, even at small amplitudes. The results look weird when the signal isn't composed of purely integer harmonics of the driving frequency, so maybe our signal isn't at integer harmonics and thats the problem, but otherwise the function works fine.

- [x] The fourier series of $\langle j(t) \rangle$ which we use to calculate $\int \,dt \langle j(t) \rangle \langle j(t + \tau) \rangle$ could be incorrect, in which case the second half of our connected correlator would be incorrect. I don't find this super likely since the connected correlator decays to zero as expected, but idk.

  - The fourier series of the current operator matches the numerically calculated current operator perfectly.

- [x] The fourier series of $\int dt\, \langle j(t) \rangle \langle j(t + \tau) \rangle$ is wrong, despite the current fourier series being correct.

  - So this may in some sense be true. The fourier series for $\langle j(t) \rangle$ is correct, but when we compare the analytically derived fourier series $\langle j(t) \rangle \langle j(t + \tau) \rangle = \sum_{n = -N}^N |j_n|^2 e^{i 2 \pi n \Omega t}$ to when we manually calculate $\langle j(t) \rangle \langle j(t + \tau) \rangle$ by numerically multiplying their individual fourier series, we find that the two differ by a *constant*.

  - The issue here is, if we assume that the manually calculated data is correct, then the connected current correlator won't go to zero. But clearly the theoretical and numerical data doesn't align, which is inherently weird.

    - I figured it out - when calculating the solution using the analytical series $\sum_{n = -N}^N | j_n |^2 e^{i 2 \pi n \Omega \tau}, we calculate this at each momentum $k$, i.e.

    $$ \langle j_k(t) \rangle \langle j_k(t + \tau) \rangle = \sum_{n = -N}^N |j_{k, n} |^2 e^{i 2 \pi n \Omega \tau}$$
    
    and then we sum up all of the momentums at the end, resulting in

    $$ \langle j(t) \rangle \langle j(t + \tau) \rangle = \sum_{n = -N}^N \left( \sum_k |j_{k, n} |^2 \right) e^{i 2\pi n \Omega \tau}$$

    However, when comparing this to the numerical solution (i.e. numerically multiplying the current expectations together to make sure our fourier series above is correct), we calculate it at the end, so our current fourier series is

    $$\langle j(t) \rangle = \sum_{n = -N}^N \left( \sum_k j_{k, n} \right) e^{i 2 \pi n \Omega t}$$

    and hence, doing the same thing we did before, we get

    $$\langle j(t) \rangle \langle j(t + \tau) \rangle = \sum_{n = -N}^N \left| \sum_k j_{k, n} \right|^2 e^{i 2 \pi n \Omega t}$$

    so the coefficients aren't the same. Clearly, the second method allows the coefficients to cancel out and have no zero frequency part, while the first method doesn't, which is why that method gives us a current product that is not oscillating around zero.

    However, the first method is the way that Denis original told me to do this (I believe), and this is the way that makes the connected current correlator go to zero, so I have to assume that this is the right method. I will have to check if this method works properly numerically as well, but honestly the fact that the connected correlator *does* go to zero with this current product is such tempting evidence that this makes sense.

# SSH Model
The relevant code for the 1-dimensional SSH model is stored within _SSHModel/_ as a package. We assume a chain with entirely real hopping amplitudes $t_1, t_2$, an infinite bulk (and
hence periodic boundary conditions), and a classical driving field $A(t) = A_0 \sin(2\pi \Omega t)$. In this section we will describe each package at a high-level (more detailed documentation
can be found within the code) and some of the relevant theory. We always work in the eigenbasis of the unperturbed Hamiltonian.

## Quantum Regression Theorem (Single- and Double-Time Correlations)
The system of equations derived for the single-time correlations is

$$
\frac{d}{dt'} \begin{pmatrix}
  \langle \sigma_- (t') \rangle \\
  \langle \sigma_+ (t') \rangle \\
  \langle \sigma_z (t') \rangle
\end{pmatrix} = \begin{pmatrix}
  -2i (|E_k| + v_z(t') ) - \frac{1}{2}\gamma_- & 0 & -i v_\pm (t') \\
  0 & 2i (|E_k| + v_z(t') ) - \frac{1}{2} \gamma_- & -i v_\pm (t') \\
  2i v_\pm (t') & 2i v_\pm (t') & -\gamma_-
\end{pmatrix} \begin{pmatrix}
  \langle \sigma_- (t') \rangle \\
  \langle \sigma_+ (t') \rangle \\
  \langle \sigma_z (t') \rangle
\end{pmatrix} + \begin{pmatrix}
     0 \\
     0 \\
     -\gamma_-
\end{pmatrix}
$$

where $E_k := t_1 + t_2e^{ik}$, $\phi_k := \arg(E_k)$, and

$$
v_z (t) = 2t_2 \sin \left(k - \phi_k - \frac{1}{2} A(t) \right) \sin \left(\frac{1}{2} A(t) \right)
$$

$$
v_\pm (t) = 2it_2 \cos \left(k - \phi_k - \frac{1}{2} A(t) \right) \sin \left(\frac{1}{2} A(t) \right)
$$

This system is solved in __CalculateSingleTimeCorrelations() in SSHModel/SSH.py. Lines 198-212 are shown below.
```
# Solves the single time solutions.
inhomPart = -self.__params.decayConstant
args = (inhomPart,)
if debug:
    T1 = np.max(self.__correlationData.tauAxisSec)
    pbar = tqdm(total=1000, unit="it")]
    args = (inhomPart, pbar, [0, (T1)/1000],)

return integrate.solve_ivp(
    t_span = np.array([0, np.max(self.__correlationData.tauAxisSec)]),
    t_eval = self.__correlationData.tauAxisSec,
    y0 = initialConditions,
    args = args,
    **odeParams
).y
```

We can see that we set the inhomPart to $-\gamma_-$, which when passed into the function that defines the system of ODEs becomes $(0, 0, -\gamma_-)$. Then, ignoring the debug progress bar lines, we see that we solve the ODE using scipy, along the points $\tau \in [0, \frac{30}{\gamma_-}]$ (i.e. 30 units of the characteristic decay time), with initial conditions $(-0.5, -0.5, 0)$, which come from transforming the initial conditions (0, 0, -1) from the sublattice basis to the eigenbasis.

Then, using the Quantum Regression Theorem, we find that

$$
\frac{d}{dt'} \begin{pmatrix}
  \langle \sigma_i(t) \sigma_- (t') \rangle \\
  \langle \sigma_i(t) \sigma_+ (t') \rangle \\
  \langle \sigma_i(t) \sigma_z (t') \rangle
\end{pmatrix} = \begin{pmatrix}
  -2i (|E_k| + v_z(t') ) - \frac{1}{2}\gamma_- & 0 & -i v_\pm (t') \\
  0 & 2i (|E_k| + v_z(t') ) - \frac{1}{2} \gamma_- & -i v_\pm (t') \\
  2i v_\pm (t') & 2i v_\pm (t') & -\gamma_-
\end{pmatrix} \begin{pmatrix}
  \langle \sigma_i(t) \sigma_- (t') \rangle \\
  \langle \sigma_i(t) \sigma_+ (t') \rangle \\
  \langle \sigma_i(t) \sigma_z (t') \rangle
\end{pmatrix} + \begin{pmatrix}
     0 \\
     0 \\
     -\gamma_- \langle \sigma_i(t) \rangle
\end{pmatrix}
$$

where $i \in \{ +, -, z \}$ and $t' = t + \tau$. The inhomogenous part changes because to obtain this system from our equations of motion, we must left multiply by $\sigma_i (t)$ before taking the expectation, so our inhomogenous part becomes $(0, 0, -\gamma_- \langle \sigma_i(t) \rangle)$. The other change is to our initial conditions. Since we take the initial condition to be $\lim_{t \to \infty} \langle \sigma_i (t) \sigma_j (t) \rangle$ (i.e. $\tau = 0$ and the system is in steady state), we can calculate our initial relations using the algebra of pauli and ladder operators. At some t,

$$
\begin{align*}
  \langle \sigma_- (t) \sigma_-(t) \rangle &= \langle 0 \rangle = 0   & \langle \sigma_+(t) \sigma_-(t) \rangle &= \frac{1}{2} ( \langle \sigma_z(t) \rangle + 1 ) \\
  \langle \sigma_- (t) \sigma_+(t) \rangle &= -\frac{1}{2} \langle \sigma_z(t) - 1 )  & \langle \sigma_+(t) \sigma_+(t) \rangle &= \langle 0 \rangle = 0 \\
  \langle \sigma_- (t) \sigma_z(t) \rangle &= \langle \sigma_-(t) \rangle   & \langle \sigma_+(t) \sigma_z(t) \rangle &= -\langle \sigma_+(t) \rangle \\
\end{align*}
$$

$$
\begin{align*}
  \langle \sigma_z (t) \sigma_-(t) \rangle &= -\langle \sigma_-(t) \rangle \\
  \langle \sigma_z (t) \sigma_+(t) \rangle &= \langle \sigma_+(t) \rangle \\
  \langle \sigma_z (t) \sigma_z (t) \rangle &= \langle 1 \rangle = 1
\end{align*}
$$

Hence, we can calculate the double-time initial conditions from the single-time correlations, $\langle \sigma_i(t) \rangle which we have already calculated. Since we are intentionally sampling at a large enough sampling frequency, we can construct the Fourier series of each of our single-time correlations. We do this in lines 161-172 of SSHModel/SSH.py.

```
# Calculates the single-time fourier expansions.
self.__correlationData.singleTimeFourier = []
numPeriods = 10
dimPeriod = self.__params.decayConstant / self.__params.drivingFreq
steadyStateMask = (steadyStateCutoff <= self.__correlationData.tauAxisDim) & (self.__correlationData.tauAxisDim <= steadyStateCutoff + numPeriods * dimPeriod)
for i in range(3):
    self.__correlationData.singleTimeFourier.append(
        Fourier(self.__params.drivingFreq,
            samples = self.__correlationData.singleTime[i][steadyStateMask],
            samplesX = self.__correlationData.tauAxisSec[steadyStateMask],
            numPeriods = numPeriods)
        )
```

We use 10 periods, and we calculate the length of one period to be $\frac{1}{\Omega}$. In order to convert this to dimensionless units, we multiply by $\gamma_-$, and so one dimensionless period is $\frac{\gamma_-}{\Omega}$. Then, we create a mask that only covers points where the values of $\tau$ that we are looking at (0 to 30 decay periods) are past the 'steady state cutoff'. In other parts of the code, the steady state cutoff is defined to be 25 decay periods, which can be confirmed by looking at the plots - everything achieves its steady state well before 25 decay periods. Then, we pass the values of $\langle \sigma_i (\tau)$ and $\tau$ for each point in the steady state into a Fourier object, which calculates the Fourier series for each single-time correlation. These can be confirmed to be accurate by using 'overplotFourier = True' when plotting the single-time correlations, and zooming in will show that the Fourier series match the numerical solutions perfectly.

Now, we have attained Fourier series of our single-time correlations. This means we have these correlations at essentially arbitrary accuracy. So, for every point t within one steady state period that we choose to use as an initial condition, we can calculate the double-time initial conditions in lines 298-317.

```
return np.array([
  # When left-multiplying by $\sigma_-(t)$
  [
    np.zeros(self.__correlationData.tAxisSec.size, dtype=complex),
    -0.5 * (self.__correlationData.singleTimeFourier[2].Evaluate(self.__correlationData.tAxisSec) - 1),
    self.__correlationData.singleTimeFourier[0].Evaluate(self.__correlationData.tAxisSec)
  ],
  # When left-multiplying by $\sigma_+(t)$
  [
    0.5 * (self.__correlationData.singleTimeFourier[2].Evaluate(self.__correlationData.tAxisSec) + 1),
    np.zeros(self.__correlationData.tAxisSec.size, dtype=complex),
    -self.__correlationData.singleTimeFourier[1].Evaluate(self.__correlationData.tAxisSec)
  ],
  # When left-multiplying by $\sigma_z(t)$
  [
    -self.__correlationData.singleTimeFourier[0].Evaluate(self.__correlationData.tAxisSec),
    self.__correlationData.singleTimeFourier[1].Evaluate(self.__correlationData.tAxisSec),
    np.ones(self.__correlationData.tAxisSec.size, dtype=complex),
  ]], dtype=complex
)
```

We calculate the initial conditions by evaluating the fourier series of each single-time correlation (indices 0 = $\sigma_-$, 1 = $\sigma_+$, and 2 = $\sigma_z$) at the desired time $t$. We do this for every time $t$ at once using vectorised numpy operations.

Now that we have changed the inhomogenous part and the initial conditions, we can solve our new system of ODEs by the QRT. In lines 231-265, we have

```
self.__CalculateTAxis(steadyStateCutoff, numT)

# Defines the double time solutions. The first dimension corresponds to the left-hand operator,
# the second corresponds to the right hand operator, the third dimension corresponds to
# the different times within a steady-state period that we consider our initial conditions at, and
# the fourth dimension corresponds to the value of our time offset $\tau$.
self.__correlationData.doubleTime = np.zeros((3, 3, self.__correlationData.tAxisSec.size, self.__correlationData.tauAxisSec.size), dtype=complex)

# Calculates the double-time initial conditions based on the single-time correlations for
# each time within the steady-state period that we want to calculate.
doubleTimeInitialConditions = self.__CalculateDoubleTimeInitialConditions()

# if debug:
#     print(f"Calculating double-time correlations...")

# Loops through each initial condition time t.
outerIterable = self.__correlationData.tAxisSec
if debug:
  outerIterable = tqdm(outerIterable)

  for tIndex, t in enumerate(outerIterable):
    # Loops through all 3 operators that we can left-multiply by.
    for i in range(3):
      # Calculates the new inhomogenous term.
        newInhomPart = -self.__params.decayConstant * self.__correlationData.singleTimeFourier[i].Evaluate(t)[0]
        args = (newInhomPart,)

        # Solves system.
        self.__correlationData.doubleTime[i, :, tIndex, :] = integrate.solve_ivp(
          t_span = t + np.array([0, np.max(self.__correlationData.tauAxisSec)]),
          t_eval = t + self.__correlationData.tauAxisSec,
          y0 = doubleTimeInitialConditions[i, :, tIndex],
          args = args,
          **odeParams
        ).y
```

The first line, given a point at which we have entered the steady state (which, as mentioned before, we consider the steady state to start at 25 decay periods), and the number of points on the t-axis within one steady state period that we want to use, will return a list of all the times $t$ that we want to use as our initial conditions. Then, the double-time initial conditions are calculated using the process described before. Then, we loop through each time $t$ and each left-hand operator $\sigma_i(t)$ and calculate the corresponding inhomogenous term $-\gamma_- \langle \sigma_i(t) \rangle$, and pass that into the system along with the corresponding initial conditions. This solves the double-time correlations at the times (where our integration variable is $t' = t + \tau$) $t' = t$ to $t' = t + \tau_{max}$. It is worth mentioning here that every time we calculate a value, we use the $t$ and $\tau$ values in seconds. Only when plotting do we use the dimensioneless parameters.

## Current Operators
The single-time current operator is calculated directly using vectorised operators using the formula

$$
j(t) = -t_2 \sin(k - \phi_k - A(t)) \sigma_z(t) + i t_2 \cos(k - \phi_k - A(t)) (\sigma_+(t) - \sigma_- (t))
$$

and then taking the expectation of both sides. In code, this is calculated in CalculateCurrent() in SSHModel/SSH.py. This function is generally self-explanatory if you choose to read it. We can also expand this formula to calculate

$$
\begin{align*}
j(t) j(t + \tau) &= t_2^2 \[ \sin(k - \phi_k - A(t)) \sin(k - \phi_k - A(t + \tau)) \sigma_z(t) \sigma_z (t + \tau) \\
&- \cos(k - \phi_k - A(t)) \cos(k - \phi_k - A(t + \tau)) ( \sigma_+(t) \sigma_+(t + \tau) - \sigma_+(t) \sigma_-(t + \tau) - \sigma_-(t) \sigma_+(t + \tau) + \sigma_-(t) \sigma_-(t + \tau) ) \\
&- i\cos (k - \phi_k - A(t)) \sin(j - \phi_k - A(t + \tau)) (\sigma_+ (t) \sigma_z(t + \tau) - \sigma_-(t) \sigma_z (t + \tau) ) \\
&- i\sin(k - \phi_k - A(t)) \cos(k - \phi_k - A(t + \tau)) (\sigma_z(t) \sigma_+(t + \tau) - \sigma_z(t) \sigma_-(t + \tau) \]
\end{align*}
$$

The final product that we want is $\frac{1}{T} \int dt\, \langle j(t) j(t + \tau) \rangle - \langle j(t) \rangle \langle j(t + \tau) \rangle$, where clearly $\frac{1}{T} = \Omega$. By using the fourier series of the current

$$
\langle j(t) \rangle = \sum_{n = -N}^N j_n e^{i 2\pi n \Omega t}
$$

we can find that

$$
\Omega \int dt\, \langle j(t) \rangle \langle j(t + \tau) \rangle = \sum_{n = -N}^N |j_n|^2 e^{i 2\pi n \Omega \tau}
$$

and this is calculated in CalculateFourier() in SSHModel/CurrentData.py. The term $\int \,dt \langle j(t) \rangle \langle j(t + \tau) \rangle$ is calculated in the later parts of CalculateCurrent() in SSHModel/SSH.py. First, __CalculateDoubleTimeCurrent() implements the formula for $\langle j(t) j(t + \tau) \rangle$ seen above. Then, we integrate it numerically along the t axis in lines 481-490.

```
# Integrates our double-time correlation function over a single steady-state period w.r.t t.
self.__currentData.integratedDoubleTimeData = self.__params.drivingFreq * np.trapezoid(
  y = self.__currentData.doubleTimeData,
  x = self.__correlationData.tAxisSec,
  axis = 0
)

# Subtracts the (already integrated) double product data from the newly calculated integrated double-time
# correlation function.
self.__currentData.doubleConnectedCorrelator = self.__currentData.integratedDoubleTimeData - self.__currentData.doubleProductData
```

The integrated data is then subtracted from the fourier series representation of the second half of the connected correlator we saw above, and that is how we calculate the desired integrated connected correlator.

The final step is to find the fourier transforms at the harmonics. This is done self-explanatorily using the integral definition

$$
\mathcal{F}[D_k(\tau)] (\omega) = \int_{-\infty}^\infty D_k(\tau') e^{-2 \pi \omega \tau'} \,d\tau'
$$

for a fixed momentum $k$, where $\omega$ is in $s^{-1}$ and we restrict the domain to be the domain of $\tau$ where it is in the steady state (typically the steady state is everything past 25 decay periods). Once this is calculated in lines 396-421 of SSHModel/SSH.py,

```
angularFreq = 2 * np.pi * self.__params.drivingFreq
# Array to store the fourier transforms.
harmonics = np.zeros((2 * maxHarmonic + 1,), dtype=complex)

# Gets the data in steady-state, we will not consider data not in steady state.
steadyStateTauAxis = self.__correlationData.tauAxisSec[steadyStateMask]
steadyStateConnectedCorrelator = self.__currentData.doubleConnectedCorrelator[steadyStateMask]

# Creates the required exponential terms. First axis is degree of harmonic, second axis is tau axis.
expTerms = np.outer(-1j * np.arange(-maxHarmonic, maxHarmonic + 1) * angularFreq, steadyStateTauAxis)
expTerms = np.exp(expTerms)

# We want to multiply each fixed n with the value of the connected correlator at every value of tauaxis.
# So, we loop through each harmonic index.
integrand = np.zeros(expTerms.shape, dtype=complex)
for nIndex in range(expTerms.shape[0]):
  integrand[nIndex, :] = expTerms[nIndex, :] * steadyStateConnectedCorrelator

  # Now, our integrand contains every relevant term $D_k(\tau) e^{-in \omega \tau}$, with the first axis determining n, and the second
  # determining $\tau$. Hence, since we are integrating along the tau axis to determine the magnitude at each harmonic, we integrate
  # along the tau axis (axis 1).
  for nIndex in range(expTerms.shape[0]):
    harmonics[nIndex] = np.trapezoid(
      y = integrand[nIndex, :],
      x = steadyStateTauAxis
    )
```

Hopefully the comments above are illuminating enough about how the integration occurs, but this is how we calculate the fourier transforms at the integer harmonics. This is the final step of our program. Of course, there are many implementation details that are not mentioned here. The code itself is generally rather well-documented, if a little messy or sloppy at times, so understanding the full process by reading the code should be an achievable process.

