import numpy as np
import matplotlib.pyplot as plt
from config.paths import DATA_DIR, PLOTTING_DIR

axes, one_current = np.load(DATA_DIR / "D=1.0, k=31.npy", allow_pickle = True)
_, three_current = np.load(DATA_DIR / "D=3.0, k=31.npy", allow_pickle = True)
maxN = 50

# g2(0) FULL SIZE
plt.plot(np.arange(1, maxN + 1), one_current.second_order_correlation_function[0], marker = 'x', label=r"$\Delta = 1$, x")
plt.plot(np.arange(1, maxN + 1), one_current.second_order_correlation_function[1], marker = 'x', label=r"$\Delta = 1$, y")
plt.plot(np.arange(1, maxN + 1), three_current.second_order_correlation_function[0], marker = 'x', label=r"$\Delta = 3$, x")
plt.plot(np.arange(1, maxN + 1), three_current.second_order_correlation_function[1], marker = 'x', label=r"$\Delta = 3$, y")
plt.legend()
plt.xlim((0, 11))
plt.xlabel(r"$\omega / \Omega$")
plt.ylabel(r"$g_2(0)$")
plt.yscale('log')
plt.savefig(PLOTTING_DIR / "g2(0).png", dpi=300)
plt.show()

# MODE NUMBER
plt.plot(np.arange(1, 3 * maxN + 1), one_current.semiclassical_mode_population[0], marker = 'x', label=r"$\Delta = 1$, x")
plt.plot(np.arange(1, 3 * maxN + 1), one_current.semiclassical_mode_population[1], marker = 'x', label=r"$\Delta = 1$, y")
plt.plot(np.arange(1, 3 * maxN + 1), three_current.semiclassical_mode_population[0], marker = 'x', label=r"$\Delta = 3$, x")
plt.plot(np.arange(1, 3 * maxN + 1), three_current.semiclassical_mode_population[1], marker = 'x', label=r"$\Delta = 3$, y")
plt.legend()
plt.xlim((0, 11))
plt.xlabel(r"$\omega / \Omega$")
plt.ylabel(r"$g_2(0)$")
plt.yscale('log')
plt.savefig(PLOTTING_DIR / "n_cl.png", dpi=300)
plt.show()

# g2(0) ZOOMED IN
# plt.plot(np.arange(1, 51), one_current.second_order_correlation_function[0], marker = 'x', label=r"$\Delta = 1$, x")
# plt.plot(np.arange(1, 51), one_current.second_order_correlation_function[1], marker = 'x', label=r"$\Delta = 1$, y")
# plt.plot(np.arange(1, 51), three_current.second_order_correlation_function[0], marker = 'x', label=r"$\Delta = 3$, x")
# plt.plot(np.arange(1, 51), three_current.second_order_correlation_function[1], marker = 'x', label=r"$\Delta = 3$, y")
# plt.legend()
# plt.xlim((0, 11))
# plt.ylim(0.9975, 1.01)
# plt.xlabel(r"$\omega / \Omega$")
# plt.ylabel(r"$g_2(0)$")
# plt.savefig(PLOTTING_DIR / "[Zoomed] g2(0).png", dpi=300)
# plt.show()