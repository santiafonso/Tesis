"""Figura del avance del KMC por tanda: donde se simulo cada punto y en que orden.

    python -m autoexp.kmc_plot_tandas --state autoexp/kmc/g-4/kmc.json [--out .../avance.png]

Fondo: reconstruccion actual con los puntos cargados. Cada punto va marcado con el numero de
su tanda (color por tanda, orden fijo); los de la tanda en curso van huecos. Los pseudo-puntos
(cota + monotonia en l) llevan borde punteado.
"""
import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from autoexp import kmc_planner

# orden fijo (colores por identidad de tanda, nunca reciclados dentro de una figura)
COLORS = ["#2a6fdb", "#e0751f", "#1f9e89", "#c23b7e", "#7a5bd6", "#8c6d1f", "#3b8f3b", "#d64545", "#4d4d4d"]
ETAPA = {1: "grilla"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    here = os.path.dirname(a.state)
    st = kmc_planner.load(a.state)
    out = a.out or os.path.join(here, "avance.png")
    lx, le = kmc_planner.axes(st)
    pseudo = set(map(tuple, st.get("pseudo", [])))
    jobs = st.get("jobs", {})

    rec, _ = kmc_planner.reconstruct(st)
    fig, ax = plt.subplots(figsize=(8.6, 7.6), constrained_layout=True)
    im = ax.imshow(rec, cmap="Greys_r", vmin=0, vmax=1, alpha=0.55,
                   extent=[le[0], le[-1], lx[-1], lx[0]], aspect="auto")
    for k, r in enumerate(st["rounds"], start=1):
        pts = [(int(x["row"]), int(x["col"])) for x in csv.DictReader(open(os.path.join(here, r["file"])))]
        info = jobs.get(str(k), {})
        en_curso = bool(info) and "done" not in info and k > 1
        names = {"serafin": "Serafín", "mulatona": "Mulatona"}
        hosts = sorted({names.get(p["host"].split("@")[1].split(".")[0], "?") for p in info.get("parts", [])}) or ["Serafín"]
        c = COLORS[(k - 1) % len(COLORS)]
        etapa = ETAPA.get(k, "bisección %d" % (k - 1) if k <= 6 else "relleno")
        lab = "Tanda %d · %s · %d pts · %s%s" % (k, etapa, len(pts), "/".join(hosts), " (en curso)" if en_curso else "")
        for n, (i, j) in enumerate(pts):
            ps = (i, j) in pseudo
            ax.scatter(le[j], lx[i], s=150, marker="o", facecolors="white" if en_curso else c, edgecolors=c,
                       linewidths=2.2, linestyle=":" if ps else "-", zorder=3, label=lab if n == 0 else None)
            ax.annotate(str(k), (le[j], lx[i]), ha="center", va="center", fontsize=7.5, fontweight="bold",
                        color=c if en_curso else "white", zorder=4)
    if st.get("largos"):
        L = np.array(st["largos"]["puntos"])
        ax.scatter(le[L[:, 1]], lx[L[:, 0]], s=420, marker="s", facecolors="none", edgecolors="#222",
                   linewidths=1.3, linestyle="--", zorder=2, label="job largo 48 h (Mulatona, en curso)")
    ax.scatter([], [], s=150, facecolors="#bbb", edgecolors="#555", linestyle=":", linewidths=2.2,
               label="borde punteado = pseudo (cota + monotonía)")
    ax.set_xlabel(r"$\log(\ell)$")
    ax.set_ylabel(r"$\log(\Xi)$")
    ax.set_title("KMC g=-4: %d de %d puntos, por tanda" % (len(st["known_d"]), st["budget"]))
    fig.colorbar(im, ax=ax, label="SoC reconstruido (fondo)", shrink=0.85)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=8, frameon=False)
    fig.savefig(out, dpi=140)
    print("->", out)


if __name__ == "__main__":
    main()
