"""Crudo 64x64 de g=-4 para los profes: mapa KMC (corregido por las filas basura del final de datos-*.dat),
la zona cortada a las 24 h recuadrada, el borde del continuo y la linea de tendencia (donde se tiraron las
transectas del 12/10), y la diferencia con el continuo.
Uso: python autoexp/kmc/g-4/grid64/figura_profes.py"""
import os, re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "..")))
from autoexp.oracle import load_truth

m = np.load(os.path.join(HERE, "final_1012", "mapa.npy")); cota = np.load(os.path.join(HERE, "final_1012", "cota.npy"))
n_fix = 0
for line in open(os.path.join(HERE, "correccion_filas_basura_1012.txt")):
    p, v = line.split()[:2]
    r, c = map(int, re.search(r"r(\d+)_c(\d+)", p).groups())
    if not cota[r, c] and abs(m[r, c] - float(v)) > 1e-6:
        m[r, c] = float(v); n_fix += 1
np.save(os.path.join(HERE, "final_1012", "mapa_corregido.npy"), m)
lx, le = np.linspace(2, -4, 64), np.linspace(-4, 2, 64)
from scipy import ndimage
lab, nl = ndimage.label(cota)
zona = lab == (1 + int(np.argmax(ndimage.sum(cota, lab, range(1, nl + 1)))))  # la zona cortada, sin pixeles sueltos
ext = [le[0], le[-1], lx[-1], lx[0]]
d = 6 / 63 / 2  # medio pixel

# continuo 128 y lineas: borde (SoC 0.5) del continuo y la tendencia KMC (acantilado corrido en log Xi,
# caida corrida en log l) con la que se eligieron las transectas
c128 = load_truth("-4.0"); X128, L128 = np.linspace(2, -4, 128), np.linspace(-4, 2, 128)
g = -4; dxi = 0.087 * g + 0.052; dl = -0.0175 * g * g + 0.016 * g + 0.038
cliff = np.array([X128[::-1][np.argmax(c128[::-1, j] > 0.5)] for j in range(128)])
fall = np.array([L128[np.where(c128[i] > 0.5)[0].max()] if (c128[i] > 0.5).any() else np.nan for i in range(128)])
from scipy.interpolate import RegularGridInterpolator
f = RegularGridInterpolator((X128[::-1], L128), c128[::-1])
XX, LL = np.meshgrid(lx, le, indexing="ij")
con64 = f(np.c_[XX.ravel(), LL.ravel()]).reshape(64, 64)
tr = np.genfromtxt(os.path.join(HERE, "..", "..", "tendencia", "tanda.csv"), delimiter=",", names=True)
tr = tr[tr["g"] == g]

def lineas(ax, leyenda=False):
    # el acantilado va hasta donde se junta con la caida; la caida, desde ahi para arriba
    jj = L128 <= np.nanmax(fall[X128 > cliff.max()]) if np.isfinite(fall).any() else L128 < 99
    top = X128 >= cliff[jj].max()
    ax.plot(L128[jj], cliff[jj], "w--", lw=1.2, label="borde del continuo" if leyenda else None)
    ax.plot(fall[top], X128[top], "w--", lw=1.2)
    ax.plot(L128[jj], cliff[jj] + dxi, color="#ff4d4d", lw=1.6, label="tendencia (transectas nuevas)" if leyenda else None)
    ax.plot(fall[top] + dl, X128[top], color="#ff4d4d", lw=1.6)
    ax.scatter(tr["logell"], tr["logxi"], s=10, marker="o", facecolors="none", edgecolors="#ff4d4d", lw=0.8,
               label="puntos de las transectas" if leyenda else None)
    # zona cortada a las 24 h: contorno de las cotas
    ax.contour(le, lx, zona.astype(float), levels=[0.5], colors="k", linewidths=1.4)
    ax.set_xlim(-4 - d, 2 + d); ax.set_ylim(-4 - d, 2 + d)

fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.6), constrained_layout=True)
im = axs[0].imshow(m, cmap="viridis", vmin=0, vmax=1, extent=ext, aspect="auto", interpolation="nearest")
axs[0].contourf(le, lx, zona.astype(float), levels=[0.5, 1.5], colors="none", hatches=["///"])
lineas(axs[0], leyenda=True)
axs[0].set_title("KMC 64×64, g = −4  (%d corridas)" % m.size)
fig.colorbar(im, ax=axs[0], label="SoC máx", shrink=0.9)
dif = np.where(cota, np.nan, m - con64)
cmd = plt.get_cmap("RdBu_r").copy(); cmd.set_bad("0.85")
im2 = axs[1].imshow(dif, cmap=cmd, vmin=-0.6, vmax=0.6, extent=ext, aspect="auto", interpolation="nearest")
lineas(axs[1])
axs[1].set_title("KMC − continuo (sin la zona cortada)")
fig.colorbar(im2, ax=axs[1], label="diferencia de SoC", shrink=0.9)
from matplotlib.patches import Patch
h, l = axs[0].get_legend_handles_labels()
h.append(Patch(facecolor="none", edgecolor="k", hatch="///")); l.append("cortadas a las 24 h (el valor es un mínimo)")
axs[0].legend(h, l, loc="lower right", fontsize=8, framealpha=0.85)
for ax in axs:
    ax.set_xlabel("log ℓ"); ax.set_ylabel("log Ξ")
out = os.path.join(HERE, "final_1012", "crudo_profes.png")
fig.savefig(out, dpi=170)
print("corregidos %d puntos; %d cotas; %d transectas en g=-4 -> %s" % (n_fix, cota.sum(), len(tr), out))
