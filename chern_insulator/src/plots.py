import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from config.paths import DATA_DIR, PLOTTING_DIR, STYLESHEET

axes, one_current = np.load(DATA_DIR / "A=0.2, D=1.0, k=31.npy", allow_pickle = True)
_, three_current =  np.load(DATA_DIR / "A=0.2, D=3.0, k=31.npy", allow_pickle = True)
maxN = 50

plt.style.use(STYLESHEET)

# g2(0) FULL SIZE
# plt.plot(np.arange(1, maxN + 1), one_current.second_order_correlation_function[0], marker = 'x', color='red', linestyle='-')
# plt.plot(np.arange(1, maxN + 1), one_current.second_order_correlation_function[1], marker = 'x', color='red', linestyle='--')
# plt.plot(np.arange(1, maxN + 1), three_current.second_order_correlation_function[0], marker = 'x', color='blue', linestyle='-')
# plt.plot(np.arange(1, maxN + 1), three_current.second_order_correlation_function[1], marker = 'x', color='blue', linestyle='--')

# color_handles = [
#     Line2D([0], [0], color='blue', linestyle='-', label='triv'),
#     Line2D([0], [0], color='red',  linestyle='-', label='top'),
# ]

# style_handles = [
#     Line2D([0], [0], color='k', linestyle='-',  label='x'),
#     Line2D([0], [0], color='k', linestyle='--', label='y'),
# ]

# legend1 = plt.legend(handles=color_handles, loc='upper left')
# plt.gca().add_artist(legend1)  # needed so the first legend isn't overwritten
# legend2 = plt.legend(handles=style_handles, loc='upper right')
# plt.gca().add_artist(legend2)

# plt.xlim((0, 9))
# plt.xlabel(r"$\omega / \Omega$")
# plt.ylabel(r"$g_2(0)$")
# # plt.yscale('log')
# plt.savefig(PLOTTING_DIR / "g2(0).png", dpi=300)
# plt.show()

# # MODE NUMBER
# plt.plot(np.arange(1, 3 * maxN + 1), one_current.semiclassical_mode_population[0], marker = 'x', color='red', linestyle='-')
# plt.plot(np.arange(1, 3 * maxN + 1), one_current.semiclassical_mode_population[1], marker = 'x', color='red', linestyle='--')
# plt.plot(np.arange(1, 3 * maxN + 1), three_current.semiclassical_mode_population[0], marker = 'x', color='blue', linestyle='-')
# plt.plot(np.arange(1, 3 * maxN + 1), three_current.semiclassical_mode_population[1], marker = 'x', color='blue', linestyle='--')

# color_handles = [
#     Line2D([0], [0], color='blue', linestyle='-', label='triv'),
#     Line2D([0], [0], color='red',  linestyle='-', label='top'),
# ]

# style_handles = [
#     Line2D([0], [0], color='k', linestyle='-',  label='x'),
#     Line2D([0], [0], color='k', linestyle='--', label='y'),
# ]

# legend1 = plt.legend(handles=color_handles, loc='upper left')
# plt.gca().add_artist(legend1)  # needed so the first legend isn't overwritten
# legend2 = plt.legend(handles=style_handles, loc='upper right')
# plt.gca().add_artist(legend2)

# plt.xlim((0, 9))
# plt.xlabel(r"$\omega / \Omega$")
# plt.ylabel(r"$n_{cl}$")
# plt.yscale('log')
# plt.savefig(PLOTTING_DIR / "n_cl.png", dpi=300)
# plt.show()

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

# delta n
# for gamma_m_index in range(one_current.dc_population_variance_weak_laser.shape[2]):
#     plt.plot(np.arange(1, maxN + 1), one_current.dc_population_variance_weak_laser[0, :, gamma_m_index], marker = 'x', color='red', linestyle='-')
#     plt.plot(np.arange(1, maxN + 1), one_current.dc_population_variance_weak_laser[1, :, gamma_m_index], marker = 'x', color='red', linestyle='--')
#     plt.plot(np.arange(1, maxN + 1), three_current.dc_population_variance_weak_laser[0, :, gamma_m_index], marker = 'x', color='blue', linestyle='-')
#     plt.plot(np.arange(1, maxN + 1), three_current.dc_population_variance_weak_laser[1, :, gamma_m_index], marker = 'x', color='blue', linestyle='--')

# color_handles = [
#     Line2D([0], [0], color='blue', linestyle='-', label='triv'),
#     Line2D([0], [0], color='red',  linestyle='-', label='top'),
# ]

# style_handles = [
#     Line2D([0], [0], color='k', linestyle='-',  label='x'),
#     Line2D([0], [0], color='k', linestyle='--', label='y'),
# ]

# legend1 = plt.legend(handles=color_handles, loc='upper left')
# plt.gca().add_artist(legend1)  # needed so the first legend isn't overwritten
# legend2 = plt.legend(handles=style_handles, loc='upper right')
# plt.gca().add_artist(legend2)

# plt.xlim((0, 9))
# plt.xlabel(r"$\omega / \Omega$")
# plt.ylabel(r"$\delta n_{\mu, m}$")
# # plt.yscale('log')
# plt.savefig(PLOTTING_DIR / "n_cl.png", dpi=300)
# plt.show()

# squeezing
# Determine the number of gamma indices to calculate the color gradients
# num_gamma = one_current.squeezing.shape[2]

# # Sample from the Reds and Blues colormaps. 
# # We start at 0.4 to ensure the lightest colors remain visible against a white background.
# red_shades = plt.cm.Reds(np.linspace(0.4, 1.0, num_gamma))
# blue_shades = plt.cm.Blues(np.linspace(0.4, 1.0, num_gamma))

# # Plot the lines using the generated shades
# for gamma_m_index in range(num_gamma):
#     current_red = red_shades[gamma_m_index]
#     current_blue = blue_shades[gamma_m_index]
    
#     plt.plot(np.arange(1, maxN + 1), one_current.squeezing[0, :, gamma_m_index], marker='x', color=current_red, linestyle='-')
#     plt.plot(np.arange(1, maxN + 1), one_current.squeezing[1, :, gamma_m_index], marker='x', color=current_red, linestyle='--')
#     plt.plot(np.arange(1, maxN + 1), three_current.squeezing[0, :, gamma_m_index], marker='x', color=current_blue, linestyle='-')
#     plt.plot(np.arange(1, maxN + 1), three_current.squeezing[1, :, gamma_m_index], marker='x', color=current_blue, linestyle='--')

# # Dynamically build the color legend handles to include the gamma_m_index
# gamma_m = np.log10(np.logspace(-4, -1, 5))
# color_handles = []
# for i in range(num_gamma):
#     color_handles.append(Line2D([0], [0], color=blue_shades[i], linestyle='-', label=rf'$\gamma_m / \Delta = 10^{{{gamma_m[i]:.2f}}}$'))
# for i in range(num_gamma):
#     color_handles.append(Line2D([0], [0], color=red_shades[i], linestyle='-', label=rf'$\gamma_m / \Delta = 10^{{{gamma_m[i]:.2f}}}$'))

# style_handles = [
#     Line2D([0], [0], color='k', linestyle='-',  label='x'),
#     Line2D([0], [0], color='k', linestyle='--', label='y'),
# ]

# top_handles = [
#     Line2D([0], [0], color='blue', linestyle='-',  label='triv'),
#     Line2D([0], [0], color='red', linestyle='-', label='top'),
# ]

# # Configure legends
# legend1 = plt.legend(handles=color_handles, loc='upper right', fontsize='small', ncol=2) 
# plt.gca().add_artist(legend1)  # needed so the first legend isn't overwritten

# legend2 = plt.legend(handles=style_handles, loc='upper right', bbox_to_anchor=(1.0, 0.825))
# plt.gca().add_artist(legend2)

# legend3 = plt.legend(handles=top_handles, loc='upper right', bbox_to_anchor=(0.94, 0.825))
# plt.gca().add_artist(legend3)

# plt.xlim((0, 9))
# plt.xlabel(r"$\omega / \Omega$")
# plt.ylabel(r"$\eta_{\mu, m}$")
# # plt.axhline(0, color='black')
# # plt.yscale('log')

# plt.savefig(PLOTTING_DIR / "squeezing_weak_laser.png", dpi=300)
# plt.show()

# Plot the lines using the generated shades 
# plt.plot(np.arange(1, maxN + 1), one_current.squeezing[0, :], marker='x', color="red", linestyle='-')
# plt.plot(np.arange(1, maxN + 1), one_current.squeezing[1, :], marker='x', color="red", linestyle='--')
# plt.plot(np.arange(1, maxN + 1), three_current.squeezing[0, :], marker='x', color="blue", linestyle='-')
# plt.plot(np.arange(1, maxN + 1), three_current.squeezing[1, :], marker='x', color="blue", linestyle='--')

# style_handles = [
#     Line2D([0], [0], color='k', linestyle='-',  label='x'),
#     Line2D([0], [0], color='k', linestyle='--', label='y'),
# ]

# top_handles = [
#     Line2D([0], [0], color='blue', linestyle='-',  label='triv'),
#     Line2D([0], [0], color='red', linestyle='-', label='top'),
# ]

# # Configure legends
# legend2 = plt.legend(handles=style_handles, loc='upper right', bbox_to_anchor=(1.0, 0.825))
# plt.gca().add_artist(legend2)

# legend3 = plt.legend(handles=top_handles, loc='upper right', bbox_to_anchor=(0.94, 0.825))
# plt.gca().add_artist(legend3)

# plt.xlim((0, 9))
# plt.xlabel(r"$\omega / \Omega$")
# plt.ylabel(r"$\eta_{\mu, m}$")
# # plt.axhline(0, color='black')
# # plt.yscale('log')

# plt.savefig(PLOTTING_DIR / "squeezing.png", dpi=300)
# plt.show()

plt.plot(np.arange(1, maxN + 1), one_current.angular_momentum, marker='x', color="red", linestyle='-')
plt.plot(np.arange(1, maxN + 1), three_current.angular_momentum, marker='x', color="blue", linestyle='-')

top_handles = [
    Line2D([0], [0], color='blue', linestyle='-',  label='triv'),
    Line2D([0], [0], color='red', linestyle='-', label='top'),
]

legend3 = plt.legend(handles=top_handles, loc='upper right', bbox_to_anchor=(0.94, 0.825))
plt.gca().add_artist(legend3)

plt.xlim((0, 9))
plt.xlabel(r"$\omega / \Omega$")
plt.ylabel(r"$L_m$")
# plt.axhline(0, color='black')
# plt.yscale('log')

plt.savefig(PLOTTING_DIR / "angular_momentum.png", dpi=300)
plt.show()