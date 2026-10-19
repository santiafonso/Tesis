"""Los 13 g en una imagen: por g, KMC reconstruido | continuo | KMC - continuo."""
import json
import os
import sys

REPO = "/home/santiafonso/Tesis-autoexp"
sys.path.insert(0, os.path.join(REPO, "presentaciones", "presentacion7_10_10"))
import figuras as F  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
F.KMC = os.path.join(HERE, "kmc")
GS = list(range(-8, 5))
PER_ROW = 3
nrow = -(-len(GS) // PER_ROW)

fig = plt.figure(figsize=(PER_ROW * 7.6, nrow * 3.2 + 0.8))
outer = fig.add_gridspec(nrow, PER_ROW, wspace=0.12, hspace=0.55, left=0.03, right=0.94, top=0.93, bottom=0.04)
im_soc = im_dif = None
for k, g in enumerate(GS):
    st = F.state(g)
    fp = os.path.join(F.gdir(g), "dip", "final.npy")
    if os.path.exists(fp):
        fin, tag = np.load(fp), "TPS+DIP"
    else:
        fin, tag = np.load(os.path.join(F.gdir(g), "mapa_kmc.npy")), "solo TPS (DIP en cola)"
    con = F.load_truth("%.1f" % g)
    cp = os.path.join(F.gdir(g), "control.json")
    ctl = ""
    c = json.load(open(cp)).get("fusion", {}) if os.path.exists(cp) and tag == "TPS+DIP" else {}
    ccsv = os.path.join(F.gdir(g), "control.csv")
    if c and c.get("n", 0) >= 20:
        ctl = "  ·  control: RMS %.3f (%.1f dB eq.)" % (c["rms"], c["psnr_equiv"])
    elif tag == "TPS+DIP" and os.path.exists(ccsv):
        ctl = "  ·  control sin terminar (detenido 16/10)"
    else:
        ctl = "  ·  esperando DIP final"
    inner = outer[k // PER_ROW, k % PER_ROW].subgridspec(1, 3, wspace=0.08)
    axs = [fig.add_subplot(inner[0, j]) for j in range(3)]
    im_soc = F._mapa(axs[0], fin, st, "KMC %s, %d medidos" % (tag, F.n_meas(st)))
    F._mapa(axs[1], con, title="continuo")
    im_dif = F._mapa(axs[2], fin - con, st, "KMC − continuo", cmap="RdBu_r", vmin=-0.5, vmax=0.5, pts=False)
    if os.path.exists(ccsv):
        cc = np.loadtxt(ccsv, delimiter=",", skiprows=1, usecols=(2, 3), ndmin=2)
        for ax in (axs[0], axs[2]):
            ax.scatter(cc[:, 1], cc[:, 0], marker="^", s=22, facecolors="#ff9f1c", edgecolors="k", linewidths=0.5)
    for ax in axs[1:]:
        ax.set_yticklabels([])
    axs[0].set_ylabel("log Ξ", fontsize=9)
    for ax in axs:
        ax.set_xlabel("log ℓ", fontsize=8, labelpad=1)
    axs[1].text(0.5, 1.16, F.gtxt(g) + ctl, transform=axs[1].transAxes, ha="center",
                fontsize=12, fontweight="bold")

cax1 = fig.add_axes([0.955, 0.55, 0.01, 0.35])
cax2 = fig.add_axes([0.955, 0.10, 0.01, 0.35])
fig.colorbar(im_soc, cax=cax1, label="SoC máx")
fig.colorbar(im_dif, cax=cax2, label="KMC − continuo")
fig.suptitle("Reconstrucción con KMC en los 13 g: mapa final, continuo y diferencia "
             "(puntos blancos = medidos, círculos rojos = pseudo, triángulos naranjas = puntos de control)\n"
             "KMC con los errores de tasas de superficie y J1 (binario viejo); estado de Mendieta al 19/10", fontsize=14, y=0.985)
out = os.path.join(HERE, "todos_finales.png")
fig.savefig(out, dpi=130)
print(out)
