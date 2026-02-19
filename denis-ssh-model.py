import numpy as np
from numpy import pi, cos, sin, sqrt, fft, exp
from scipy.integrate import simpson, solve_ivp
from scipy.special import jv
import matplotlib
from matplotlib import rc
from tqdm import tqdm

rc('font', **{'family': 'serif', 'serif': ['Times']})
rc('text', usetex=True)

matplotlib.use('macosx')


class One_D_SSH_Model:
    def __init__(self, omega=2.0 / 3.01, amp=0.2, nc=9, number_points_k=251, t1=2.0, t2=1.0, dt=0.010122):
        self.t1 = t1  # t_1 parameter of 1D SSH
        self.t2 = t2  # t_2 parameter of 1D SSH
        self.omega = omega  # \omega carrying frequency of pumping
        self.amp = amp  # amplitude A of pumping envelope
        self.gamma = 0.1  # phonon scattering time
        self.num = 11  # max harmonic number in fourier expansions and such.

        # Physical system constants, not used directly in most functions.
        self.NN = 1e8  # number of emitters
        self.beta0 = 1.0  # g_0N
        self.Qc = 100  # cavity quality factor

        self.dt = dt  # time step
        self.nc = nc  # number of cycles in pumping envelope
        self.t_max = 2.0 * nc * pi / omega  # one pumping period times number of desired pumping periods.
        # Generates array from 0 to max, determined by number of pumping periods.
        self.time = np.linspace(0.0, self.t_max, int(self.t_max / self.dt), endpoint=True)

        # Generates 21 points within the first pumping period.
        self.time_inf = np.linspace(0.0, 2.0 * pi / omega, 21, endpoint=True)
        # Multiplies fftfreq frequencies by the period in (s), so divides by the frequency in Hz.
        # Hence, normalises the frequencies to units of harmonics of the driving frequency.
        self.freq = pi * 2.0 * fft.fftfreq(self.time.size, self.dt) / self.omega  # normalized Fourier frequency#
        # Function to calculate driving term at times x.
        self.Ax = lambda x: amp * sin(omega * x)  # * (sin(omega * x / (2.0 * nc)) ** 2)  # if x <= self.t_max else 0.0
        # pumping envelope
        self.number_points_k = number_points_k  # number of k-points
        self.number_points_t = self.time.size  # number of time-points
        # Sampling first BZ.
        self.KK = np.linspace(-pi, pi, number_points_k, endpoint=True)  # k-grid
        # range of KK / num points = k-step.
        self.dk = (self.KK[-1] - self.KK[0]) / number_points_k  # k-step

        # Initial density matrix, ground state of lower band.
        self.rho_thermal = np.array([[0, 0], [0, 1]], dtype=complex)

    def integration_over_momentum(self, func):
        """ Takes the integration over the 1st Br zone """
        # Divides by 2pi to get the average over the zone.
        return simpson(func, self.KK, axis=-1) / (2.0 * pi)  # simpson integration

    def energy(self, k: float):
        """ Energy band for the given momentum k"""
        return sqrt(self.t2 ** 2 + self.t1 ** 2 + 2.0 * self.t2 * self.t1 * cos(k))

    def A12(self, k: float):
        """ Berry connection for the given momentum k
        Called A_12 because its the off-diagonal element, at row 1, col 2."""
        return -0.5 * self.t2 * (self.t2 + self.t1 * cos(k)) / (self.energy(k) ** 2)

    def dwdk(self, k: float):
        """ Derivative of the energy band for the given momentum k"""
        return -self.t2 * self.t1 * sin(k) / self.energy(k)

    def hz(self, k: float, t: float):
        """Value of $H_z$ at momentum k, time t."""
        dwdk = self.dwdk(k)
        At = self.Ax(t)
        ener = self.energy(k)
        dA12 = 2.0 * ener * self.A12(k)
        return ener - dwdk * sin(At) + dA12 * (1.0 - cos(At))

    def hy(self, k: float, t: float):
        """Value of $H_y$ at momentum k, time t."""
        dwdk = self.dwdk(k)
        dA12 = 2.0 * self.energy(k) * self.A12(k)
        At = self.Ax(t)
        return dwdk * (1.0 - cos(At)) + dA12 * sin(At)

    def hzn(self, n: int, k: float):
        """Fourier coefficient $h_{z, n}$ for momentum k."""
        dwdk = self.dwdk(k)
        dA12 = 2.0 * self.energy(k) * self.A12(k)
        if n == 0:
            res = self.energy(k) + dA12 * (1.0 - jv(n, self.amp))
        elif abs(n) % 2 == 1:
            res = 1.0j * dwdk * jv(n, self.amp)
        else:
            res = -dA12 * jv(n, self.amp)
        return res

    def hyn(self, n: int, k: float):
        """Fourier coefficient $h_{y, n}$ for momentum k."""
        dwdk = self.dwdk(k)
        dA12 = 2.0 * self.energy(k) * self.A12(k)
        if n == 0:
            res = dwdk * (1.0 - jv(n, self.amp))
        elif abs(n) % 2 == 1:
            res = -1.0j * dA12 * jv(n, self.amp)
        else:
            res = -dwdk * jv(n, self.amp)
        return res

    def jz(self, k: float, t: float):
        """Value of j_z at momentum k, time t."""
        dwdk = self.dwdk(k)
        At = self.Ax(t)
        ener = self.energy(k)
        dA12 = 2.0 * ener * self.A12(k)
        return dwdk * cos(At) - dA12 * sin(At)

    def jy(self, k: float, t: float):
        """Value of $j_y$ at momentum k, time t."""
        dwdk = self.dwdk(k)
        dA12 = 2.0 * self.energy(k) * self.A12(k)
        At = self.Ax(t)
        return dwdk * sin(At) + dA12 * cos(At)

    def jzn(self, n: int, k: float):
        """Fourier coefficient $n$ for $j_z$ at momentum k."""
        if abs(n) % 2 == 1:
            res = -1.0j * 2.0 * self.energy(k) * self.A12(k) * jv(n, self.amp)
        else:
            res = self.dwdk(k) * jv(n, self.amp)
        return res

    def jyn(self, n: int, k: float):
        """Fourier coefficient $n$ for $j_y$ at momentum k."""
        if abs(n) % 2 == 1:
            res = 1.0j * self.dwdk(k) * jv(n, self.amp)
        else:
            res = 2.0 * self.energy(k) * self.A12(k) * jv(n, self.amp)
        return res

    def to_dict(self, mat):
        """Given an array mat, which contains -num to num, we get a dictionary that we can index directly
        as -num to num."""
        return {n: mat[n + self.num] for n in range(-self.num, self.num + 1)}

    def to_array(self, mat):
        """Turns dict from last function into array."""
        return np.array([mat[n] for n in range(-self.num, self.num + 1)])

    def to_time_array(self, mat: dict, time: np.array):
        """Coefficient list mat, time array time, turns it into a fourier series evaluation."""
        num = self.num
        res = np.zeros(len(time), dtype=complex)
        # At each time, result is coefficient matrix mat times corresponds exponentials
        # $e^{i \omega n t}$. So, mat is coefficient array and using time, this turns it into a function.
        for ind, t in enumerate(time):
            res[ind] = np.sum([mat[n] * exp(1.0j * self.omega * n * t) for n in range(-num, num + 1)], axis=0)
        return res

    def build_conv_matrix(self, kernel):
        """
        Build an (2N+1)x(2N+1) matrix C such that
        (C @ v)[n] = sum_m kernel[m] * v[n-m]
        with zero padding outside [-N, N].

        Matrix so that if kernel and v are both fourier coefficients,
        forming C from kernel and doing C @ v calculates fourier coefficients
        for product of fourier series, up to coefficients -n to n.
        """
        kernel = np.asarray(kernel, dtype=np.complex128)
        M = kernel.size
        # M = 2N + 1
        N = (M - 1) // 2
        # C is MxM
        C = np.zeros((M, M), dtype=np.complex128)
        
        # Loop through values up to 2N + 1
        for n_idx in range(M):
            # n is between -n, n.
            n = n_idx - N
            for k_idx in range(M):
                # Similarly, k is between -n and n.
                k = k_idx - N
                m = n - k
                # We know n is between -N and N.
                # Make sure n - k is between -N and N
                if -N <= m <= N:
                    # If so, matrix entry n, k is kernel_m
                    C[n_idx, k_idx] = kernel[m + N]
        return C

    def sigma_t(self, k, time_range=None) -> tuple[np.array]:
        # Gets $\sigma_-, \sigma_+, \sigma_z$ at times in time_range.
        if time_range is None:
            time_range = self.time

        num = self.num # Maximum harmonic number.
        M = 2 * num + 1
        # Thought of as 3 arrays of size M, which is 2n + 1.
        # Used to somehow calculate coefficients of fourier series
        # for sigma operators.
        b = np.zeros(3 * M, dtype=complex)
        # Sets the third fourier series to represent -2$\gamma$.
        b[2 * M + num] -= 2.0 * self.gamma
        eig, U, invU = self.quasi_energy(k)
        # Takes inverse of the diagonal eigenvalue matrix
        # by taking reciprocal of eigenvalues.
        D_inv = np.diag(1.0 / eig)
        sigma = U @ D_inv @ invU @ b

        # Evalates first, second, and third 'row' of sigma as fourier coefficients in a fourier series, over time axis.
        # These are our $\sigma_-, \sigma_+, \sigma_z$.
        sigma_m = self.to_time_array(self.to_dict(sigma[0:M]), time_range)
        sigma_p = self.to_time_array(self.to_dict(sigma[M:2 * M]), time_range)
        sigma_z = self.to_time_array(self.to_dict(sigma[2 * M:3 * M]), time_range)

        return sigma_m, sigma_p, sigma_z

    def quasi_energy(self, k):
        num = self.num
        # Contains fourier coefficients for hy and hz,.
        hy = np.array([self.hyn(n, k) for n in range(-num, num + 1)], dtype=complex)
        hz = np.array([self.hzn(n, k) for n in range(-num, num + 1)], dtype=complex)
        # Calculates $i\Omega n + 2\gamma_-$ for all n.
        a = np.array([1.0j * self.omega * n + 2.0 * self.gamma for n in range(-num, num + 1)], dtype=complex)
        # Calculates $i\Omega n + \gamma$.
        b = np.array([1.0j * self.omega * n + self.gamma for n in range(-num, num + 1)], dtype=complex)

        M = 2 * num + 1

        # Builds the convolution matrices for coefficients for $h_y$ and $h_z$.
        Hy = self.build_conv_matrix(hy)
        Hz = self.build_conv_matrix(hz)

        # Build 3M × 3M system
        A = np.zeros((3 * M, 3 * M), dtype=np.complex128)
        V = np.zeros((3 * M, 3 * M), dtype=np.complex128)
        # --- Equation (1) block ---
        # (-2 Hy)x + (-2 Hy)y + diag(a) z = RHS
        A[0:M, 0:M] = 2 * Hy
        A[0:M, M:2 * M] = 2 * Hy
        A[0:M, 2 * M:3 * M] = np.diag(a)

        V[2 * M:3 * M, 0:M] = np.eye(M)
        V[M:2 * M, M:2 * M] = np.eye(M)
        V[0:M, 2 * M:3 * M] = np.eye(M)

        # --- Equation (2) block ---
        # 0*x + (diag(b) - 2i Hz) y + (-Hy) z = 0
        A[M:2 * M, M:2 * M] = np.diag(b) - 2j * Hz
        A[M:2 * M, 2 * M:3 * M] = -Hy

        # --- Equation (3) block ---
        # (diag(b) + 2i Hz)x + 0*y + (-Hy)z = 0
        A[2 * M:3 * M, 0:M] = np.diag(b) + 2j * Hz
        A[2 * M:3 * M, 2 * M:3 * M] = -Hy

        # Gets eigenvalues and unitary matrix U.
        eig, U = np.linalg.eig(V @ A)
        # Gets $U^{-1}$.
        invU = np.linalg.inv(U)

        return eig, U, invU

    def evolution(self, k: float, sys0=None, bt=1.0):
        """ Calculates the evolution of density matrix for the given momentum k,
                initial state rho_0 and time range time_range"""
        if sys0 is None:
            """Sublattice basis or eigenbasis? Aren't these initial conditions for the sublattice basis?"""
            sys0 = np.array([0, 0, -1.0], dtype=complex)

        def A_t(t):
            """Matrix for system of ODEs."""
            hz = self.hz(k, t)
            hy = self.hy(k, t)
            return np.array([
                [-(self.gamma + 2.0j * hz), 0.0, hy],
                [0.0, -(self.gamma - 2.0j * hz), hy],
                [-2.0 * hy, -2.0 * hy, - 2.0 * self.gamma]
            ], dtype=complex)

        def system(t, y):
            """System of ODEs."""
            dy = A_t(t) @ y + 2.0 * self.gamma * np.array([0, 0, -bt], dtype=complex)
            return dy

        sys = solve_ivp(fun=system, t_span=(0.0, self.time[-1]), y0=sys0, t_eval=self.time,
                        method='RK45', rtol=1e-9, atol=1e-12).y
        return sys

    def vec(self, X):
        return X.reshape((-1,), order='F')

    def make_matrix(self, v):
        return v.reshape((2, 2), order='F')

    def comm_blocks(self, k):
        comm_blocks = {}
        num = self.num
        n_list = np.arange(-num, num + 1)
        I_d = np.eye(2, dtype=complex)
        for m in n_list:
            Hm = self.Hn(m, k).astype(complex)
            comm_blocks[m] = 1.0j * (np.kron(I_d, Hm) - np.kron(Hm.T, I_d))
        return comm_blocks

    def average_current(self, k: float):
        sm0, sp0, sz0 = self.sigma_t(k)
        jz = self.jz(k, self.time)
        jy = self.jy(k, self.time)
        return sz0 * jz + 1.0j * jy * (sp0 - sm0)

    def correlator(self, k: float, order='direct'):
        """Calculates connected double-time current correlator."""
        # Axis along which we are solving.
        time = self.time
        # First dimension is number of points in driving period, the other is number of points
        # in time axis (tAxis, tauAxis).
        correlation = np.zeros((len(self.time_inf), len(time)), dtype=complex)
        # Gets the values of our single-time expectations (in steady state, since these
        # are calculated using fourier series).
        sm0, sp0, sz0 = self.sigma_t(k)

        for ind, t_inf in enumerate(tqdm(self.time_inf)):
            # For each time in a steady state period, evaluates
            # every product of current coefficients at times t and t + tau.
            # The functions jz and jy, which are the current coefficients,
            # use the values of the single-time expectations in the steady state.
            jz_jz = self.jz(k, time + t_inf) * self.jz(k, t_inf)
            jy_jy = self.jy(k, time + t_inf) * self.jy(k, t_inf)
            jy_jz = self.jy(k, time + t_inf) * self.jz(k, t_inf)
            jz_jy = self.jz(k, time + t_inf) * self.jy(k, t_inf)

            # First part finds percentile of the last driving cycle, multiplied by
            # length of time points to find approximate time index that
            # last driving cycle starts at.

            # Second part finds the number of time points per steady state point, i.e. how many time points we have to
            # move forward to reach the next steady-state point. Multiplies by the current steady-state index to determine
            # how many indices in time array we have to move forward to get to *current* steady-state index.

            # Hence, finds the index in the time array corresponding to the current steady-state initial condition.
            index0 = (self.nc - 1) * len(self.time) // self.nc + ind * len(self.time) // (self.nc * len(self.time_inf))

            if order == 'direct':  # <j(t+dt)j(t)>
                sigmam = np.array([0.0, (sz0[index0] + 1.0) / 2.0, -sm0[index0]], dtype=complex)
                sigmap = np.array([(1.0 - sz0[index0]) / 2.0, 0.0, sp0[index0]], dtype=complex)
                sigmaz = np.array([sm0[index0], -sp0[index0], 1.0], dtype=complex)
            else:  # <j(t)j(t+dt)>
                # The initial conditions for each left operator are evaluated using the single-time correlations
                # at the steady-state initial condition index.

                # $\sigma_- \sigma_- = 0, \sigma_- \sigma_+ = (1 - \sigma_z) / 2, \sigma_- \sigma_z = \sigma_-$
                sigmam = np.array([0.0, (1.0 - sz0[index0]) / 2.0, sm0[index0]], dtype=complex)
                # $\sigma_+ \sigma_- = (1 + \sigma_z) / 2, \sigma_+ \sigma_+ = 0, \sigma_+ \sigma_z = -\sigma_+$
                sigmap = np.array([(1.0 + sz0[index0]) / 2.0, 0.0, -sp0[index0]], dtype=complex)
                # $\sigma_z \sigma_- = -\sigma_-, \sigma_z \sigma_+ = \sigma_+, \sigma_z \sigma_z = 1$
                sigmaz = np.array([-sm0[index0], sp0[index0], 1.0], dtype=complex)

                # The above initial conditions match my theoretical derivations.

            # Calculates evolution, using current momentum, initial conditions for left-hand operator sigma_i,
            # and inhomogenous part -2 $\gamma$ times the value of the left-hand operator at the initial conditions.

            # Labels assume right multiplication, but we are only considering left multiplication, so swap
            # the labels in your mind,.
            sm_sm, sp_sm, sz_sm = self.evolution(k, sigmam, sm0[index0])
            sm_sp, sp_sp, sz_sp = self.evolution(k, sigmap, sp0[index0])
            sm_sz, sp_sz, sz_sz = self.evolution(k, sigmaz, sz0[index0])

            correlation[ind] = jz_jz * (sz_sz - sz0[index0] * sz0)
            correlation[ind] += jy_jy * (
                    (sm_sp - sp0[index0] * sm0) + (sp_sm - sm0[index0] * sp0) - (sm_sm - sm0[index0] * sm0) - (
                    sp_sp - sp0[index0] * sp0))
            correlation[ind] -= 1.0j * jz_jy * (sz_sm - sm0[index0] * sz0 - (sz_sp - sp0[index0] * sz0))
            correlation[ind] -= 1.0j * jy_jz * (sm_sz - sz0[index0] * sm0 - (sp_sz - sz0[index0] * sp0))

        return correlation

    def integration_over_tau(self, func, n):
        """ Takes the integration over time for frequency n Omega """
        return simpson(func * exp(1.0j * n * self.omega * self.time), self.time)

    def integration_over_period(self, func):
        """Integrated over the steady state period and divides by the period."""
        return self.omega * simpson(func, self.time_inf, axis=0) / (2 * pi)  # simpson integration 
