"""Adelanto del crudo 64x64 de g=-4 con los dos arreglos (refresco de superficie + `- J1`), con lo
terminado al 19/10 a la manana, contra el crudo viejo (`grid64/final_lector/mapa_corregido.npy`).

Uso: python autoexp/kmc/g-4/grid64_refresco/adelanto_1910/adelanto.py
"""
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
VIEJO = os.path.join(AQUI, "..", "..", "grid64", "final_lector")
RES = 64
lx = np.linspace(2, -4, RES)       # filas: log Xi de 2 (arriba) a -4
le = np.linspace(-4, 2, RES)       # columnas: log l


def main():
    old = np.load(os.path.join(VIEJO, "mapa_corregido.npy"))
    cont = np.load(os.path.join(VIEJO, "continuo.npy"))
    new = np.full((RES, RES), np.nan)
    cota = np.zeros((RES, RES), bool)
    for p in sorted(glob.glob(os.path.join(AQUI, "res_p*.csv"))):
        for r in csv.DictReader(open(p)):
            if r["status"] in ("terminado", "cota") and r["value"]:
                i, j = int(r["row"]), int(r["col"])
                new[i, j] = float(r["value"])
                cota[i, j] = r["status"] == "cota"
    term = np.isfinite(new) & ~cota
    d = np.where(term, new - old, np.nan)

    ext = [le[0], le[-1], lx[-1], lx[0]]
    kw = dict(extent=ext, aspect="auto", interpolation="nearest")
    cm = plt.get_cmap("viridis").copy()
    cm.set_bad("0.85")
    fig, axs = plt.subplots(1, 4, figsize=(22, 5.2), constrained_layout=True)

    im = axs[0].imshow(old, cmap=cm, vmin=0, vmax=1, **kw)
    axs[0].set_title("KMC viejo (con los dos errores)")
    fig.colorbar(im, ax=axs[0], label="SoC max")

    axs[1].imshow(np.ma.masked_invalid(np.where(term, new, np.nan)), cmap=cm, vmin=0, vmax=1, **kw)
    ii, jj = np.nonzero(cota)
    axs[1].scatter(le[jj], lx[ii], c=new[ii, jj], cmap=cm, vmin=0, vmax=1, marker="^", s=14,
                   edgecolors="k", linewidths=0.3, label="en curso (cota, SoC parcial)")
    axs[1].legend(loc="lower right", fontsize=8)
    axs[1].set_title("KMC corregido: %d terminados + %d en curso" % (term.sum(), cota.sum()))
    fig.colorbar(im, ax=axs[1], label="SoC max")

    cmd = plt.get_cmap("RdBu_r").copy()
    cmd.set_bad("0.85")
    im = axs[2].imshow(np.ma.masked_invalid(d), cmap=cmd, vmin=-1, vmax=1, **kw)
    nz = term & ~((new < 0.01) & (old < 0.01))
    axs[2].set_title("corregido - viejo, solo terminados\n(%d con SoC > 0; %d dan 0 en los dos)"
                     % (nz.sum(), term.sum() - nz.sum()))
    fig.colorbar(im, ax=axs[2], label="diferencia de SoC")
    for ax in axs[:3]:
        ax.set_xlabel("log l")
        ax.set_ylabel("log Xi")
        ax.set_xlim(-3.2, 2)

    # transecta: la columna con mas puntos terminados
    j = int(np.argmax(term.sum(0)))
    a = axs[3]
    a.plot(lx, cont[:, j], color="0.55", ls="--", lw=2, label="continuo")
    a.plot(lx, old[:, j], color="#d95f02", lw=2, marker="o", ms=4, label="KMC viejo")
    t = term[:, j]
    a.plot(lx[t], new[t, j], color="#1b9e77", lw=2, marker="o", ms=6, label="KMC corregido (terminado)")
    c = cota[:, j]
    if c.any():
        a.scatter(lx[c], new[c, j], color="#1b9e77", marker="^", s=40, facecolors="none",
                  label="corregido en curso (cota)")
    a.set_xlabel("log Xi")
    a.set_ylabel("SoC max")
    a.set_ylim(-0.03, 1.05)
    a.set_title("Corte en log l = %.2f" % le[j])
    a.grid(alpha=0.3)
    a.legend(fontsize=8, loc="center left")
    for ax in axs[:3]:
        ax.axvline(le[j], color="w", lw=0.8, ls=":")

    fig.suptitle("g = -4: adelanto del crudo 64x64 con el KMC corregido (19/10, ~5 % terminado)")
    out = os.path.join(AQUI, "adelanto.png")
    fig.savefig(out, dpi=150)
    nzd = d[nz]
    print("terminados %d (no triviales %d), cotas %d; corregido-viejo media %+.3f, RMS %.3f -> %s"
          % (term.sum(), nz.sum(), cota.sum(), nzd.mean(), np.sqrt((nzd ** 2).mean()), out))
    print("RMS vs continuo (no triviales): viejo %.3f, corregido %.3f"
          % (np.sqrt(((old - cont)[nz] ** 2).mean()), np.sqrt(((new - cont)[nz] ** 2).mean())))


if __name__ == "__main__":
    main()
