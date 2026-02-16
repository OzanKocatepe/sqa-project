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
        self.num = 11  # max harmonic number
        self.NN = 1e8  # number of emitters
        self.beta0 = 1.0  # g_0N
        self.Qc = 100  # cavity quality factor

        self.dt = dt  # time step
        self.nc = nc  # number of cycles in pumping envelope
        self.t_max = 2.0 * nc * pi / omega  # period of pumping
        self.time = np.linspace(0.0, self.t_max, int(self.t_max / self.dt), endpoint=True)

        self.time_inf = np.linspace(0.0, 2.0 * pi / omega, 21, endpoint=True)
        self.freq = pi * 2.0 * fft.fftfreq(self.time.size, self.dt) / self.omega  # normalized Fourier frequency
        self.Ax = lambda x: amp * sin(omega * x)  # * (sin(omega * x / (2.0 * nc)) ** 2)  # if x <= self.t_max else 0.0
        # pumping envelope
        self.number_points_k = number_points_k  # number of k-points
        self.number_points_t = self.time.size  # number of time-points
        self.KK = np.linspace(-pi, pi, number_points_k, endpoint=True)  # k-grid
        self.dk = (self.KK[-1] - self.KK[0]) / number_points_k  # k-step

        self.rho_thermal = np.array([[0, 0], [0, 1]], dtype=complex)

    def integration_over_momentum(self, func):
        """ Takes the integration over the 1st Br zone """
        return simpson(func, self.KK, axis=-1) / (2.0 * pi)  # simpson integration

    def energy(self, k: float):
        """ Energy band for the given momentum k"""
        return sqrt(self.t2 ** 2 + self.t1 ** 2 + 2.0 * self.t2 * self.t1 * cos(k))

    def A12(self, k: float):
        """ Berry connection for the given momentum k"""
        return -0.5 * self.t2 * (self.t2 + self.t1 * cos(k)) / (self.energy(k) ** 2)

    def dwdk(self, k: float):
        """ Derivative of the energy band for the given momentum k"""
        return -self.t2 * self.t1 * sin(k) / self.energy(k)

    def hz(self, k: float, t: float):
        dwdk = self.dwdk(k)
        At = self.Ax(t)
        ener = self.energy(k)
        dA12 = 2.0 * ener * self.A12(k)
        return ener - dwdk * sin(At) + dA12 * (1.0 - cos(At))

    def hy(self, k: float, t: float):
        dwdk = self.dwdk(k)
        dA12 = 2.0 * self.energy(k) * self.A12(k)
        At = self.Ax(t)
        return dwdk * (1.0 - cos(At)) + dA12 * sin(At)

    def hzn(self, n: int, k: float):
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
        dwdk = self.dwdk(k)
        At = self.Ax(t)
        ener = self.energy(k)
        dA12 = 2.0 * ener * self.A12(k)
        return dwdk * cos(At) - dA12 * sin(At)

    def jy(self, k: float, t: float):
        dwdk = self.dwdk(k)
        dA12 = 2.0 * self.energy(k) * self.A12(k)
        At = self.Ax(t)
        return dwdk * sin(At) + dA12 * cos(At)

    def jzn(self, n: int, k: float):
        if abs(n) % 2 == 1:
            res = -1.0j * 2.0 * self.energy(k) * self.A12(k) * jv(n, self.amp)
        else:
            res = self.dwdk(k) * jv(n, self.amp)
        return res

    def jyn(self, n: int, k: float):
        if abs(n) % 2 == 1:
            res = 1.0j * self.dwdk(k) * jv(n, self.amp)
        else:
            res = 2.0 * self.energy(k) * self.A12(k) * jv(n, self.amp)
        return res

    def to_dict(self, mat):
        return {n: mat[n + self.num] for n in range(-self.num, self.num + 1)}

    def to_array(self, mat):
        return np.array([mat[n] for n in range(-self.num, self.num + 1)])

    def to_time_array(self, mat: dict, time: np.array):
        num = self.num
        res = np.zeros(len(time), dtype=complex)
        for ind, t in enumerate(time):
            res[ind] = np.sum([mat[n] * exp(1.0j * self.omega * n * t) for n in range(-num, num + 1)], axis=0)
        return res

    def build_conv_matrix(self, kernel):
        """
        Build an (2N+1)x(2N+1) matrix C such that
        (C @ v)[n] = sum_m kernel[m] * v[n-m]
        with zero padding outside [-N, N].
        """
        kernel = np.asarray(kernel, dtype=np.complex128)
        M = kernel.size
        N = (M - 1) // 2
        C = np.zeros((M, M), dtype=np.complex128)

        for n_idx in range(M):
            n = n_idx - N
            for k_idx in range(M):
                k = k_idx - N
                m = n - k
                if -N <= m <= N:
                    C[n_idx, k_idx] = kernel[m + N]
        return C

    def sigma_t(self, k, time_range=None) -> tuple[np.array]:
        if time_range is None:
            time_range = self.time
        num = self.num
        M = 2 * num + 1
        b = np.zeros(3 * M, dtype=complex)
        b[2 * M + num] -= 2.0 * self.gamma
        eig, U, invU = self.quasi_energy(k)
        D_inv = np.diag(1.0 / eig)
        sigma = U @ D_inv @ invU @ b

        sigma_m = self.to_time_array(self.to_dict(sigma[0:M]), time_range)
        sigma_p = self.to_time_array(self.to_dict(sigma[M:2 * M]), time_range)
        sigma_z = self.to_time_array(self.to_dict(sigma[2 * M:3 * M]), time_range)

        return sigma_m, sigma_p, sigma_z

    def quasi_energy(self, k):
        num = self.num
        hy = np.array([self.hyn(n, k) for n in range(-num, num + 1)], dtype=complex)
        hz = np.array([self.hzn(n, k) for n in range(-num, num + 1)], dtype=complex)
        a = np.array([1.0j * self.omega * n + 2.0 * self.gamma for n in range(-num, num + 1)], dtype=complex)
        b = np.array([1.0j * self.omega * n + self.gamma for n in range(-num, num + 1)], dtype=complex)

        M = 2 * num + 1

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

        eig, U = np.linalg.eig(V @ A)
        invU = np.linalg.inv(U)

        return eig, U, invU

    def evolution(self, k: float, sys0=None, bt=1.0):
        """ Calculates the evolution of density matrix for the given momentum k,
                initial state rho_0 and time range time_range"""
        if sys0 is None:
            sys0 = np.array([0, 0, -1.0], dtype=complex)

        def A_t(t):
            hz = self.hz(k, t)
            hy = self.hy(k, t)
            return np.array([
                [-(self.gamma + 2.0j * hz), 0.0, hy],
                [0.0, -(self.gamma - 2.0j * hz), hy],
                [-2.0 * hy, -2.0 * hy, - 2.0 * self.gamma]
            ], dtype=complex)

        def system(t, y):
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
        time = self.time
        correlation = np.zeros((len(self.time_inf), len(time)), dtype=complex)
        sm0, sp0, sz0 = self.sigma_t(k)

        for ind, t_inf in enumerate(tqdm(self.time_inf)):
            jz_jz = self.jz(k, time + t_inf) * self.jz(k, t_inf)
            jy_jy = self.jy(k, time + t_inf) * self.jy(k, t_inf)
            jy_jz = self.jy(k, time + t_inf) * self.jz(k, t_inf)
            jz_jy = self.jz(k, time + t_inf) * self.jy(k, t_inf)

            index0 = (self.nc - 1) * len(self.time) // self.nc + ind * len(self.time) // (self.nc * len(self.time_inf))

            if order == 'direct':  # <j(t+dt)j(t)>
                sigmam = np.array([0.0, (sz0[index0] + 1.0) / 2.0, -sm0[index0]], dtype=complex)
                sigmap = np.array([(1.0 - sz0[index0]) / 2.0, 0.0, sp0[index0]], dtype=complex)
                sigmaz = np.array([sm0[index0], -sp0[index0], 1.0], dtype=complex)
            else:  # <j(t)j(t+dt)>
                sigmam = np.array([0.0, (1.0 - sz0[index0]) / 2.0, sm0[index0]], dtype=complex)
                sigmap = np.array([(1.0 + sz0[index0]) / 2.0, 0.0, -sp0[index0]], dtype=complex)
                sigmaz = np.array([-sm0[index0], sp0[index0], 1.0], dtype=complex)

            sm_sm, sp_sm, sz_sm = self.evolution(k, sigmam, sm0[index0])
            sm_sp, sp_sp, sz_sp = self.evolution(k, sigmap, sp0[index0])
            sm_sz, sp_sz, sz_sz = self.evolution(k, sigmaz, sz0[index0])

            correlation[ind] = jz_jz * (sz_sz - sz0[index0] * sz0)
            correlation[ind] += jy_jy * (
                    sm_sp - sp0[index0] * sm0 + sp_sm - sm0[index0] * sp0 - (sm_sm - sm0[index0] * sm0) - (
                    sp_sp - sp0[index0] * sp0))
            correlation[ind] -= 1.0j * jz_jy * (sz_sm - sm0[index0] * sz0 - (sz_sp - sp0[index0] * sz0))
            correlation[ind] -= 1.0j * jy_jz * (sm_sz - sz0[index0] * sm0 - (sp_sz - sz0[index0] * sp0))

        return correlation

    def integration_over_tau(self, func, n):
        """ Takes the integration over time for frequency n Omega """
        return simpson(func * exp(1.0j * n * self.omega * self.time), self.time)

    def integration_over_period(self, func):
        """ Takes the integration over time for frequency n Omega """
        return self.omega * simpson(func, self.time_inf, axis=0) / (2 * pi)  # simpson integration
