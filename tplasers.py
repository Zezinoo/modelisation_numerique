import numpy as np
import matplotlib.pyplot as plt

# Paramètres
Vpi = 390.0
V0 = Vpi - Vpi/2
V02 = Vpi
Vm = 10.0
fm = 100e3
wm = 2 * np.pi * fm

# Axe du temps : 5 périodes
T = 1 / fm
t = np.linspace(0, 5 * T, 4000)

# Signaux
Vref_ac = Vm * np.sin(wm * t)
Im1 = 0.5 * (1 - np.cos(np.pi * (V0 + Vref_ac) / Vpi))
Im2 = 0.5 * (1 - np.cos(np.pi * (V02 + Vref_ac) / Vpi))

# Aligner Im2 sur le même offset DC que Im1
Im2_shifted = Im2 - Im2.mean() + Im1.mean()

# Tracé
fig, ax1 = plt.subplots(figsize=(9, 5))
ax2 = ax1.twinx()

ax1.plot(t * 1e6, Vref_ac, label=r"$V_m \sin(\omega_m t)$")
ax2.plot(t * 1e6, Im1, label=r"$V_0 = V_{\pi}/2$", color="orange", linewidth=2)
ax2.plot(t * 1e6, Im2_shifted, label=r"$V_0 = 30$", color="red", linewidth=2)

ax1.set_xlabel("Temps (µs)", fontsize=14)
ax1.set_ylabel("Amplitude de la référence AC (V)", fontsize=14)
ax2.set_ylabel("Intensité optique (normalisée)", fontsize=14)

ax2.tick_params(axis='y', labelcolor='black')
ax1.tick_params(axis='both', labelsize=12)
ax2.tick_params(axis='y', labelsize=12)

ax1.set_ylim(-12, 12)

margin = 0.01
ax2.set_ylim(min(Im1.min(), Im2_shifted.min()) - margin,
             max(Im1.max(), Im2_shifted.max()) + margin)

ax1.set_title(r"Comparaison entre l'amplitude", fontsize=15)
ax1.grid(True)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=12)

fig.subplots_adjust(left=0.10, right=0.88, top=0.90, bottom=0.13)
plt.show()