from Hamiltonian import Hamiltonian
from data import Fourier, ModelParameters
import numpy as np
import matplotlib.pyplot as plt

params = ModelParameters(
    kx = np.pi / 4,
    ky = -np.pi / 8,
    delta = 1,
    drivingAmp = 0.3,
    drivingFreq = 2 / 3.01 * 1 / (2 * np.pi),
    decayConstant = 0.2,
    maxN = 25
)

h = Hamiltonian(params)

time = np.linspace(0, 10, 100)

# plt.plot(time, h.hx(time), color='black', label='hx')
# plt.plot(time,
#          Fourier(
#             freq = params.drivingFreq,
#             coeffs = h.hxn(np.arange(-params.maxN, params.maxN + 1))
#          ).Evaluate(time),
#          color='blue', linestyle='dashed', label='hxn')
# plt.legend()
# plt.show()

# plt.plot(time, np.stack((h.hy(),) * time.size), color='black', label='hy')
# plt.plot(time,
#          Fourier(
#             freq = params.drivingFreq,
#             coeffs = h.hyn(np.arange(-params.maxN, params.maxN + 1))
#          ).Evaluate(time),
#          color='blue', linestyle='dashed', label='hyn')
# plt.legend()
# plt.show()

# print(h.hzn(0), h.hz(0))

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

# plt.plot(time, h.Hm(time), color='black', label='Hm')
# plt.plot(time,
#          Fourier(
#             freq = params.drivingFreq,
#             coeffs = h.Hmn(np.arange(-params.maxN, params.maxN + 1))
#          ).Evaluate(time),
#          color='blue', linestyle='dashed', label='Hmn')
# plt.legend()
# plt.show()

# plt.plot(time, h.Hp(time), color='black', label='Hp')
# plt.plot(time,
#          Fourier(
#             freq = params.drivingFreq,
#             coeffs = h.Hpn(np.arange(-params.maxN, params.maxN + 1))
#          ).Evaluate(time),
#          color='blue', linestyle='dashed', label='Hpn')
# plt.legend()
# plt.show()

# plt.plot(time, h.Hz(time), color='black', label='Hz')
# plt.plot(time,
#          Fourier(
#             freq = params.drivingFreq,
#             coeffs = h.Hzn(np.arange(-params.maxN, params.maxN + 1))
#          ).Evaluate(time),
#          color='blue', linestyle='dashed', label='Hzn')
# plt.legend()
# plt.show()