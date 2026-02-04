# SQA Project
This repo contains the code I used to model resonance fluorescence in a variety of systems. Currently, we can model a qubit in a waveguide (following [Kocabas et. al., 2012](https://arxiv.org/abs/1111.7315))
and are currently in the process of modelling a one-dimensional Su–Schrieffer–Heeger (SSH) model with an infinite bulk.

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

The final product that we want is 
