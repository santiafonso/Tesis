"""Solo el crudo 64x64 de g=-4 con la zona cortada a las 24 h marcada (para los profes).
Usa final_1012/mapa_corregido.npy (de figura_profes.py) y cota.npy."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy import ndimage
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "final_1012")
m = np.load(os.path.join(D, "mapa_corregido.npy")); cota = np.load(os.path.join(D, "cota.npy"))
lab, nl = ndimage.label(cota)
zona = lab == (1 + int(np.argmax(ndimage.sum(cota, lab, range(1, nl + 1)))))
lx, le = np.linspace(2, -4, 64), np.linspace(-4, 2, 64)
d = 6 / 63 / 2
fig, ax = plt.subplots(figsize=(7, 5.6), constrained_layout=True)
im = ax.imshow(m, cmap="viridis", vmin=0, vmax=1, extent=[le[0] - d, le[-1] + d, lx[-1] - d, lx[0] + d],
               aspect="auto", interpolation="nearest")
ax.contourf(le, lx, zona.astype(float), levels=[0.5, 1.5], colors="none", hatches=["///"])
ax.contour(le, lx, zona.astype(float), levels=[0.5], colors="k", linewidths=1.4)
ax.legend([Patch(facecolor="none", edgecolor="k", hatch="///")], ["cortadas a las 24 h (el valor es un mínimo)"],
          loc="lower right", fontsize=9, framealpha=0.85)
ax.set_title("KMC 64×64, g = −4")
ax.set_xlabel("log ℓ"); ax.set_ylabel("log Ξ")
fig.colorbar(im, ax=ax, label="SoC máx")
out = os.path.join(D, "crudo_solo.png"); fig.savefig(out, dpi=170); print(out)
