"""Reconstruccion de g=-4 con 96 puntos KMC (dip/final.npy, 128x128) contra la grilla cruda 64x64 como verdad:
error en todos los puntos terminados de la grilla. Figura: crudo | reconstruccion (en la grilla 64) | diferencia.
Uso: python autoexp/kmc/g-4/grid64/vs_reconstruccion.py <carpeta con mapa.npy y cota.npy> <final.npy> <kmc.json>"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

d, fin_p, st_p = sys.argv[1:4]
m = np.load(os.path.join(d, "mapa.npy")); cota = np.load(os.path.join(d, "cota.npy"))
fin = np.clip(np.load(fin_p), 0, 1)
lx128, le128 = np.linspace(2, -4, 128), np.linspace(-4, 2, 128)
lx, le = np.linspace(2, -4, 64), np.linspace(-4, 2, 64)
f = RegularGridInterpolator((lx128[::-1], le128), fin[::-1])
X, L = np.meshgrid(lx, le, indexing="ij")
rec = f(np.c_[X.ravel(), L.ravel()]).reshape(64, 64)
ok = np.isfinite(m) & ~cota
e = rec - m
rms = float(np.sqrt(np.mean(e[ok] ** 2))); mae = float(np.mean(np.abs(e[ok])))
psnr = 20 * np.log10(1.0 / rms)
# franja anomala del KMC: SoC ~ 0 en log Xi alto con l bajo, donde el continuo da ~1
anom = (m < 0.05) & (X > 0.4) & (L < -1.2)  # zona anomala: SoC ~ 0 arriba a la izquierda (continuo ~1)
r_in = float(np.sqrt(np.mean(e[ok & ~anom] ** 2)))
# cotas: la reconstruccion no deberia quedar por debajo
viol = cota & (rec < m - 0.05)
st = json.load(open(st_p)); ps = set(map(tuple, st.get("pseudo", [])))
ij = np.array([list(map(int, k.split(","))) for k in st["known"]]); med = np.array([tuple(p) not in ps for p in ij])
res = dict(n_terminados=int(ok.sum()), rms=rms, mae=mae, psnr_equiv=psnr, rms_sin_franja_anomala=r_in,
           psnr_sin_franja=20 * np.log10(1 / r_in), cotas_violadas=int(viol.sum()), n_cotas=int(cota.sum()),
           puntos_reconstruccion=int(med.sum()))
json.dump(res, open(os.path.join(d, "vs_reconstruccion.json"), "w"), indent=1)
print(res)

ext = [le[0], le[-1], lx[-1], lx[0]]
fig, axs = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
cm = plt.get_cmap("viridis").copy(); cm.set_bad("0.75")
kw = dict(extent=ext, aspect="auto", interpolation="nearest")
im = axs[0].imshow(np.ma.masked_invalid(np.where(cota, np.nan, m)), cmap=cm, vmin=0, vmax=1, **kw)
ii, jj = np.nonzero(cota)
axs[0].scatter(le[jj], lx[ii], marker="^", s=6, c="w", lw=0)
axs[0].set_title("KMC crudo 64×64 (%d terminados, %d cotas)" % (ok.sum(), cota.sum()))
axs[1].imshow(rec, cmap="viridis", vmin=0, vmax=1, **kw)
axs[1].scatter(le128[ij[med, 1]], lx128[ij[med, 0]], s=6, c="w", edgecolors="k", linewidths=0.3)
axs[1].set_title("Reconstrucción con %d puntos KMC (TPS + DIP)" % med.sum())
fig.colorbar(im, ax=axs[:2], label="SoC máx", shrink=0.9)
cmd = plt.get_cmap("RdBu_r").copy(); cmd.set_bad("0.75")
im2 = axs[2].imshow(np.ma.masked_where(~ok, e), cmap=cmd, vmin=-0.5, vmax=0.5, **kw)
axs[2].set_title("reconstrucción − crudo: RMS %.3f (%.1f dB eq.)\nsin la franja anómala: RMS %.3f (%.1f dB)" % (rms, psnr, r_in, res["psnr_sin_franja"]))
fig.colorbar(im2, ax=axs[2], label="diferencia de SoC")
for ax in axs: ax.set_xlabel("log ℓ"); ax.set_ylabel("log Ξ")
out = os.path.join(d, "vs_reconstruccion.png"); fig.savefig(out, dpi=150); print(out)
