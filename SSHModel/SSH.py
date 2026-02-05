import numpy as np
import scipy.integrate as integrate
from tqdm import tqdm

from .CorrelationData import CorrelationData
from .CurrentData import CurrentData
from .Fourier import Fourier
from .SSHParameters import ModelParameters

class SSH:
    """
    Contains the core physics and calculations for the 1-dimensional SSH model.
    Since we consider a fixed momentum, this system simplifies to essentially a two-level system.
    """

    def __init__(self, params: ModelParameters):
        """
        Constructs an instance of the SSH model.

        Parameters
        ----------
        params: ModelParameters
            The parameters of the SSH model.
        """

        self.__params = params
        self.__correlationData: CorrelationData = None
        self.__currentData: CurrentData = None

    def __SinusoidalDrivingTerm(self, t: float | np.ndarray[float]) ->  float | np.ndarray[float]:
        """A classical, sinusoidal driving term for the system.
        
        Parameters
        ----------
        t : float | ndarray[float]
            The points in time (in seconds) to evaluate the driving term at.
            
        Returns
        -------
        float | ndarray[float]
            The value of the driving term at each time t.
        """

        return self.__params.drivingAmplitude * np.sin(2 * np.pi * self.__params.drivingFreq * t)

    def __ClassicallyDrivenSSHEquations(self, t: float, c: np.ndarray[complex], inhomPart: float, pbar=None, state=None) -> np.ndarray[float]:
        r"""
        The ODE (equations of motion) for the single-time expectations of $\sigma_-(t)$, $\sigma_+(t)$, and $\sigma_z(t)$. 

        Parameters
        ----------
        t : float
            The time t, in seconds.
        c : ndarray[float]
            The 3-component single-time correlation array.
        inhomPart : float
            The inhomogenous part of the system, which changes depending on the
            order of the correlation function. The correct inhomogenous parts are determined
            outside of this function.

        Returns:
        --------
        ndarray[float] 
            The value of dc/dt at some time t.
        """

        # PROGRESS BAR CODE TAKEN FROM https://stackoverflow.com/questions/59047892/how-to-monitor-the-process-of-scipy-odeint
        if pbar is not None and state is not None:
            # state is a list containing last updated time t:
            # state = [last_t, dt]
            # I used a list because its values can be carried between function
            # calls throughout the ODE integration
            last_t, dt = state
    
            # let's subdivide t_span into 1000 parts
            # call update(n) here where n = (t - last_t) / dt
            n = int((t - last_t)/dt)
            pbar.update(n)
    
            # we need this to take into account that n is a rounded number.
            state[0] = last_t + dt * n

        # Defines the useful parameters.
        A = self.__SinusoidalDrivingTerm
        vZ = 2 * self.__params.t2 * np.sin(self.__params.k - self.__params.phiK - 0.5 * A(t)) * np.sin(0.5 * A(t))
        vPm = 2j * self.__params.t2 * np.cos(self.__params.k - self.__params.phiK - 0.5 * A(t)) * np.sin(0.5 * A(t))

        # Defines the coefficient matrix.
        B = np.array([
            [
                -2j * (np.abs(self.__params.Ek) + vZ) - 0.5 * self.__params.decayConstant,
                0,
                -1j * vPm
            ],
            [
                0,
                2j * (np.abs(self.__params.Ek) + vZ) - 0.5 * self.__params.decayConstant,
                -1j * vPm
            ],
            [
                2j * vPm,
                2j * vPm,
                -self.__params.decayConstant
            ]], dtype=complex)
    
        # Inhomogenous part.
        d = np.array([0, 0, inhomPart], dtype=complex)
    
        return B @ c + d

    def Solve(self, tauAxis: np.ndarray[float], initialConditions: np.ndarray[complex], numT: int, steadyStateCutoff: float=25, debug: bool=False) -> CorrelationData:
        r"""
        Solves the system of ODEs for the expectations of our two-level system operators $(\langle \sigma_-(t) \rangle,\, \langle \sigma_+(t) \rangle,\, \langle \sigma_z(t) \rangle)$, and the double-time correlation functions $\langle \sigma_i (t) \sigma_j (t + \tau) \rangle$.
            
        Parameters
        ----------
        tauAxis : ndarray[float]
            The offsets from the initial time $t$ that the functions should be calculated on, in units of $\gamma_-^{-1}$.
       initialConditions : ndarray[complex]
            The initial conditions of the single-time correlation functions. This is used to calculate
            the initial conditions of the double-time correlations.
        numT : int
            The number of points within a steady-state period to use for the double-time correlation initial conditions.
        steadyStateCutoff : float
            The time, in units of $\gamma_-^{-1}$, after which we consider the system in steady-state. This is used
            to calculate the initial conditions for the double-time correlations.
        debug : bool
            Whether to print out debug progress statements.

        Returns
        -------
        CorrelationData
            A CorrelationData instance that contains all of the relevant information about the correlations.
        """

        # Ensures initial conditions are a numpy array.
        initialConditions = np.array(initialConditions, dtype=complex)

        self.__correlationData = CorrelationData(
            tauAxisDim = tauAxis,
            tauAxisSec = tauAxis / self.__params.decayConstant
        )
        self.__currentData = CurrentData(
            tauAxisDim = tauAxis,
            tauAxisSec = tauAxis / self.__params.decayConstant
        )

        # Stores the function parameters, since they are mostly the same for the single- and double-time
        # correlations.
        odeParams = {
            'fun' : self.__ClassicallyDrivenSSHEquations,
            'rtol' : 1e-10,
            'atol' : 1e-12,
            'max_step' : 0.01 / self.__params.decayConstant
        }

        # if debug:
        #     print("Calculating single-time correlations...")
        self.__correlationData.singleTime = self.__CalculateSingleTimeCorrelations(initialConditions, odeParams, debug=False)

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

        self.__CalculateDoubleTimeCorrelations(steadyStateCutoff, numT, odeParams, debug)
  
        return self.__correlationData
    
    def __CalculateSingleTimeCorrelations(self, initialConditions: np.ndarray[complex], odeParams: dict, debug: bool=False) -> np.ndarray[complex]:
        r"""
        Calculates the single-time correlation functions.

        Parameters
        ----------
        initialConditions : ndarray[complex]
            The initial conditions of the single-time correlation functions.
        odeParams: dict
            A dictionary containing the relevant parameters for the function which solves the ODE.
        debug : bool
            Whether to output debug progress messages.

        Returns
        -------
        ndarray[complex]
            An array of shape (3, tauAxis.size) which contains the value of the correlation functions at each time.
            The indices correspond to 0 = $\langle \sigma_-(t) \rangle$, 1 = $\langle \sigma_+(t) \rangle$, 2 = $\langle \sigma_z(t) \rangle$.
        """
 
        # Solves the single time solutions.
        inhomPart = -self.__params.decayConstant
        args = (inhomPart,)
        if debug:
            T1 = np.max(self.__correlationData.tauAxisSec)
            pbar = tqdm(total=1000, unit="it")
            args = (inhomPart, pbar, [0, (T1)/1000],)

        return integrate.solve_ivp(
            t_span = np.array([0, np.max(self.__correlationData.tauAxisSec)]),
            t_eval = self.__correlationData.tauAxisSec,
            y0 = initialConditions,
            args = args,
            **odeParams
        ).y
    
    def __CalculateDoubleTimeCorrelations(self, steadyStateCutoff: int, numT: int, odeParams: dict, debug: bool=False) -> None:
        r"""
        Calculates the double-time correlation functions and saves them to CorrelationData without returning them.

        Parameters
        ----------
        steadyStateCutoff : int
            The point in time, in units if $\gamma_-^{-1}$, after which we consider the system to be in the steady state.
        numT : int
            The number of points within a steady state period to use as the initial conditions for a set of
            double-time correlation functions.
        odeParams: dict
            A dictionary containing the relevant parameters for the function which solves the ODE.
        debug : bool
            Whether to print out debug progress statements.
        """

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

    def __CalculateTAxis(self, steadyStateCutoff: int, numT: int) -> None:
        r"""
        Calculates the points along the t axis, within a steady state period, that will
        be used when calculating the double-time correlations.
        
        Parameters
        ----------
        steadyStateCutoff : int
            The point in time, in units if $\gamma_-^{-1}$, after which we consider the system to be in the steady state.
        numT : int
            The number of points within a steady state period to use as the initial conditions for a set of
            double-time correlation functions.
        """

        startPoint = steadyStateCutoff / self.__params.decayConstant
        endPoint = steadyStateCutoff / self.__params.decayConstant + 1 / self.__params.drivingFreq
        self.__correlationData.tAxisSec = np.linspace(startPoint, endPoint, numT)
        self.__correlationData.tAxisDim = self.__correlationData.tAxisSec * self.__params.decayConstant

        self.__currentData.tAxisSec = self.__correlationData.tAxisSec
        self.__currentData.tAxisDim = self.__correlationData.tAxisDim

    def __CalculateDoubleTimeInitialConditions(self) -> np.ndarray[complex]:
        """
        Calculates the double-time initial conditions for all 9 double-time correlators, at each time t.

        Returns
        -------
        ndarray[complex]
            An array of shape (3, 3, tAxisSec.size), where the first two dimensions correspond to the left and right operator
            in the double-time correlations, and the third corresponds to which point in the steady-state period along the
            t-axis we have chosen as our initial condition.
        """

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
         
    def CalculateCurrent(self, steadyStateCutoff: float=None) -> CurrentData:
        r"""Calculates the current operator for the given parameters.
        
        Parameters
        ----------
        steadyStateCutoff : float
            The time (in units of $\gamma_-^{-1}$) which we consider the system to be in steady-state.
            i.e. we only consider the Fourier transform of the system after this point.

        Returns
        -------
        CurrentData
            The instance of CurrentData which contains all of the relevant current information.

        Raises
        ------
        ValueError
            If Solve() hasn't been called yet, so the expectation values of our operators haven't been calculated.
            This makes it impossible to calculate the current operator.
        """

        # Defines useful terms.
        drivingSamples = self.__SinusoidalDrivingTerm(self.__correlationData.tauAxisSec)

        # Calculates the current operator in terms of the previously calculated expectation values.
        self.__currentData.timeDomainData = self.__params.t2 * (
            -np.sin(self.__params.k - self.__params.phiK - drivingSamples) * self.__correlationData.singleTime[2]
            + 1j * np.cos(self.__params.k - self.__params.phiK - drivingSamples) * (self.__correlationData.singleTime[1] - self.__correlationData.singleTime[0])
        )

        if steadyStateCutoff is not None:
            mask = self.__correlationData.tauAxisDim >= steadyStateCutoff
        else:
            mask = np.full(self.__correlationData.tauAxisSec.size, True, dtype=bool)


        # Calculates the Fourier transform of the solution.
        self.__currentData.freqDomainData = np.fft.fftshift(np.fft.fft(self.__currentData.timeDomainData[mask]))

        # Calculates the relevant frequency axis.
        steadyStateAxis = self.__correlationData.tauAxisSec[mask]
        sampleSpacing = (np.max(steadyStateAxis) - np.min(steadyStateAxis)) / steadyStateAxis.size
        self.__currentData.freqAxis = np.fft.fftshift(np.fft.fftfreq(steadyStateAxis.size, sampleSpacing))

        # Calculates the fourier expansions of the current data.
        self.__currentData.CalculateFourier(self.__params.k, self.__params, self.__correlationData)

        # Calculates the double-time current data.
        self.__CalculateDoubleTimeCurrent()

        # Calculates the double product data (the second term in the connected correlator).
        self.__CalculateDoubleProductCurrent()

        # Calculates the connected correlator.
        self.__CalculateIntegratedConnectedCorrelator()

        # Calculates the FFT of the real part of the current connected correlator. Since we are using the same tau axis and steady
        # state cutoff, we can use the same frequency axis as before.
        self.__currentData.doubleConnectedCorrelatorFreqDomain = np.fft.fftshift(np.fft.fft(self.__currentData.doubleConnectedCorrelator[mask]))

        # Now that we have the connected correlator, we can also calculate the manual
        # fourier transform at the harmonics.
        self.__ManuallyCalculateFourierAtHarmonics(mask)

        return self.__currentData
    
    def __ManuallyCalculateFourierAtHarmonics(self, steadyStateMask: np.ndarray[bool] | np.ndarray[int], maxHarmonic: int=12) -> None:
        """Manually calculates the fourier transform at the harmonic frequencies.
        
        Parameters
        ----------
        steadyStateMask : ndarray[bool] | ndarray[int]
            A mask determining which points we consider in the steady state.
        maxHarmonic : int
            Determines the range of harmonics we calculate. Ranges from -maxHarmonic to maxHarmonic.
        """

        angularFreq = 2 * np.pi * self.__params.drivingFreq
        # Array to store the fourier transforms.
        harmonics = np.zeros((2 * maxHarmonic + 1,), dtype=complex)

        steadyStateMask = np.arange(self.__currentData.doubleConnectedCorrelator.size)

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

        self.__currentData.harmonics = harmonics
 
    def __CalculateDoubleProductCurrent(self) -> None:
        r"""
        Calculates the current double-time product, of the form $\int dt\, \langle j(t) \rangle \langle j(t + \tau) \rangle$, using the fact that
        it can be expressed as the fourier series $\sum_{n = -N}^N | j_n |^2 e^{i \omega n \tau}$. Stores the result directly in currentData.

        The calculated result is *after* integrating w.r.t. dt over a period.
        """

        coefficients = np.abs(self.__currentData.fourierExpansion.coeffs)**2

        # Stores the new fourier data so that we can evaluate it.
        fourier = Fourier(
            baseFreq = self.__params.drivingFreq,
            coeffs = coefficients
        )

        # Creates the manual data.
        manualData = np.zeros((self.__currentData.tAxisDim.size, self.__currentData.tauAxisDim.size), dtype=complex)
        for tIndex, t in enumerate(self.__currentData.tAxisSec):
            manualData[tIndex, :] = self.__currentData.fourierExpansion.Evaluate(t) * self.__currentData.fourierExpansion.Evaluate(t + self.__currentData.tauAxisSec)

        self.currentData._integratedManualData = self.__params.drivingFreq * np.trapezoid(
            y = manualData,
            x = self.__currentData.tAxisSec,
            axis = 0
        )
 
        self.__currentData.doubleProductData = fourier.Evaluate(self.__correlationData.tauAxisSec)
    
    def __CalculateDoubleTimeCurrent(self) -> None:
        r"""
        Calculates the double-time current correlation data in the form of $\langle j(t) j(t + \tau) \rangle$ for the current momentum.
        Stores the values in currentData directly.
        """

        self.__currentData.doubleTimeData = np.zeros((self.__correlationData.tAxisSec.size, self.__correlationData.tauAxisSec.size), dtype=complex)

        # Defines useful properties.
        kSubtract = self.__params.k - self.__params.phiK
        drivingSamplesT = self.__SinusoidalDrivingTerm(self.__correlationData.tAxisSec)

        # Calculates the current operator at all times tau for each time t.
        for tIndex, t in enumerate(self.__correlationData.tAxisSec):
            doubleTimeAtT = self.__correlationData.doubleTime[:, :, tIndex, :]
            drivingSamplesTau = self.__SinusoidalDrivingTerm(t + self.__correlationData.tauAxisSec)

            coeff1 = np.sin(kSubtract - drivingSamplesT[tIndex]) * np.sin(kSubtract - drivingSamplesTau)
            coeff2 = -np.cos(kSubtract - drivingSamplesT[tIndex]) * np.cos(kSubtract - drivingSamplesTau)
            coeff3 = -1j * np.cos(kSubtract - drivingSamplesT[tIndex]) * np.sin(kSubtract - drivingSamplesTau)
            coeff4 = -1j * np.sin(kSubtract - drivingSamplesT[tIndex]) * np.cos(kSubtract - drivingSamplesTau)

            operators1 = doubleTimeAtT[2, 2, :]
            operators2 = doubleTimeAtT[1, 1, :] - doubleTimeAtT[1, 0, :] - doubleTimeAtT[0, 1, :] + doubleTimeAtT[0, 0, :]
            operators3 = doubleTimeAtT[1, 2, :] - doubleTimeAtT[0, 2, :]
            operators4 = doubleTimeAtT[2, 1, :] - doubleTimeAtT[2, 0, :]

            self.__currentData.doubleTimeData[tIndex, :] = self.__params.t2**2 * (coeff1 * operators1 + coeff2 * operators2 + coeff3 * operators3 + coeff4 * operators4)
    
    def __CalculateIntegratedConnectedCorrelator(self) -> None:
        """Calculates the double-time connected correlator which has been integrated over one period w.r.t t.
        
        Returns
        -------
        ndarray[complex]
            An array containing the value of the integrated connected correlated at each time t + tau.
        """
        
        # Integrates our double-time correlation function over a single steady-state period w.r.t t.
        self.__currentData.integratedDoubleTimeData = self.__params.drivingFreq * np.trapezoid(
            y = self.__currentData.doubleTimeData,
            x = self.__correlationData.tAxisSec,
            axis = 0
        )

        # Subtracts the (already integrated) double product data from the newly calculated integrated double-time
        # correlation function.
        self.__currentData.doubleConnectedCorrelator = self.__currentData.integratedDoubleTimeData - self.__currentData.doubleProductData

    @property
    def correlationData(self) -> CorrelationData:
        if self.__correlationData is None:
            raise ValueError("Run Solve() first.")
        else:
            return self.__correlationData

    @property
    def currentData(self) -> CurrentData:
        if self.__currentData is None:
            raise ValueError("Run CalculateCurrent() first.")
        else:
            return self.__currentData