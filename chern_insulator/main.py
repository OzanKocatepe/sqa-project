import numpy as np
import matplotlib.pyplot as plt

from data import EnsembleParameters
from Ensemble import Ensemble

plt.style.use('stylesheet.mplstyle')

# Number of TOTAL momentum points.
numK = 75
tauMax = 50

params = EnsembleParameters(
    delta = 1,
    drivingAmp = 0.2,
    decayConstant = 0.2,
    maxN = 50
)

ensemble = Ensemble(params)
ensemble.SampleBrillouinZone(numK)
ensemble.Run(tauMax)

plt.semilogy(
    ensemble.axes.freqAxis,
    np.abs(np.fft.fftshift(np.fft.fft(ensemble.totalCurrent.paramagneticCurrent[0]))),
    label='Jx',
    color='tab:blue')

plt.semilogy(
    ensemble.axes.freqAxis,
    np.abs(np.fft.fftshift(np.fft.fft(ensemble.totalCurrent.paramagneticCurrent[1]))),
    label='Jy',
    color='orange')

plt.xlim(-11, 11)
plt.xlabel(r"$f / \Omega$")
plt.ylabel("Magnitude of FFT")
plt.legend()

for n in np.arange(-10, 11):
    plt.axvline(n, color='black', linestyle='dashed', alpha=0.2)

plt.tight_layout()
plt.show()