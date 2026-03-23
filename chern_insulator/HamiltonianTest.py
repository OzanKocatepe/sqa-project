from Hamiltonian import Hamiltonian
from data import Fourier, ModelParameters
import numpy as np
import matplotlib.pyplot as plt

params = ModelParameters(
    kx = np.pi / 4.01,
    ky = -np.pi / 8.07,
    delta = 1,
    drivingAmp = 0.3,
    decayConstant = 0.2,
    maxN = 25
)

h = Hamiltonian(params)
time = np.linspace(0, 10, 100)

def compare_hx_to_hxn() -> None:
    """Compares the hx function to the hxn fourier series over time."""
    plt.plot(time, h.hx(time), color='black', label='hx')
    plt.plot(time,
            Fourier(
                freq = params.drivingFreq,
                coeffs = h.hxn(np.arange(-params.maxN, params.maxN + 1))
            ).Evaluate(time),
            color='blue', linestyle='dashed', label='hxn')
    plt.legend()
    plt.show()

def compare_hy_to_hyn() -> None:
    """Compares the hy function to the hyn fourier series over time."""
    plt.plot(time, np.stack((h.hy(),) * time.size), color='black', label='hy')
    plt.plot(time,
            Fourier(
                freq = params.drivingFreq,
                coeffs = h.hyn(np.arange(-params.maxN, params.maxN + 1))
            ).Evaluate(time),
            color='blue', linestyle='dashed', label='hyn')
    plt.legend()
    plt.show()

def compare_hz_to_hzn() -> None:
    """Compares the hz function to the hzn fourier series over time."""
    plt.plot(time, h.hz(time), color='black', label='hz')
    plt.plot(time,
            Fourier(
                freq = params.drivingFreq,
                coeffs = h.hzn(np.arange(-params.maxN, params.maxN + 1))
            ).Evaluate(time),
            color='blue', linestyle='dashed', label='hzn')
    # plt.axhline(h.hz(0))
    # plt.axhline(h.hzn(0))
    plt.legend()
    plt.show()

def compare_Hm_to_Hmn() -> None:
    """Compares the Hm function to the Hmn fourier series over time."""
    plt.plot(time, h.Hm(time).real, color='black', label='Hm')
    plt.plot(time,
            Fourier(
                freq = params.drivingFreq,
                coeffs = h.Hmn(np.arange(-params.maxN, params.maxN + 1))
            ).Evaluate(time).real,
            color='blue', linestyle='dashed', label='Hmn')
    plt.legend()
    plt.show()

    plt.plot(time, h.Hm(time).imag, color='black', label='Hm')
    plt.plot(time,
            Fourier(
                freq = params.drivingFreq,
                coeffs = h.Hmn(np.arange(-params.maxN, params.maxN + 1))
            ).Evaluate(time).imag,
            color='blue', linestyle='dashed', label='Hmn')
    plt.legend()
    plt.show()

def compare_Hp_to_Hpn() -> None:
    """Compares the Hp function to the Hpn fourier series over time."""
    plt.plot(time, h.Hp(time).real, color='black', label='Hp')
    plt.plot(time,
            Fourier(
                freq = params.drivingFreq,
                coeffs = h.Hpn(np.arange(-params.maxN, params.maxN + 1))
            ).Evaluate(time).real,
            color='blue', linestyle='dashed', label='Hpn')
    plt.legend()
    plt.show()

    plt.plot(time, h.Hp(time).imag, color='black', label='Hp')
    plt.plot(time,
            Fourier(
                freq = params.drivingFreq,
                coeffs = h.Hpn(np.arange(-params.maxN, params.maxN + 1))
            ).Evaluate(time).imag,
            color='blue', linestyle='dashed', label='Hpn')
    plt.legend()
    plt.show()

def compare_Hz_to_Hzn() -> None:
    """Compares the Hz function to the Hzn fourier series over time."""
    plt.plot(time, h.Hz(time).real, color='black', label='Hz')
    plt.plot(time,
            Fourier(
                freq = params.drivingFreq,
                coeffs = h.Hzn(np.arange(-params.maxN, params.maxN + 1))
            ).Evaluate(time).real,
            color='blue', linestyle='dashed', label='Hzn')
    plt.legend()
    plt.show()

    plt.plot(time, h.Hz(time).imag, color='black', label='Hz')
    plt.plot(time,
            Fourier(
                freq = params.drivingFreq,
                coeffs = h.Hzn(np.arange(-params.maxN, params.maxN + 1))
            ).Evaluate(time).imag,
            color='blue', linestyle='dashed', label='Hzn')
    plt.legend()
    plt.show()