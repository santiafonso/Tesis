"""Figura del estado del KMC real: puntos corridos sobre el mapa del continuo + reconstruccion
con lo que hay hasta ahora.

Uso:
    python -m autoexp.kmc_plot --state autoexp/kmc/g-4/kmc.json [--runs corridas.csv] [--g -4.0]
                               [--out autoexp/kmc/g-4/estado.png]
Panel izquierdo: mapa del continuo (referencia, NO es la verdad del KMC) con cada punto pintado
por su SoC de KMC (circulo = medido, triangulo = pseudo-punto por cota/monotonia).
Panel derecho: reconstruccion del planificador con los puntos cargados.
"""
import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from autoexp import kmc_planner
from autoexp.oracle import load_truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--g", default="-4.0")
    ap.add_argument("--out")
    a = ap.parse_args()
    st = kmc_planner.load(a.state)
    out = a.out or a.state.replace("kmc.json", "estado.png")
    lx, le = kmc_planner.axes(st)
    ext = [le[0], le[-1], lx[-1], lx[0]]
    pseudo = set(map(tuple, st.get("pseudo", [])))
    ij = list(st["known_d"].keys())
    meas = [p for p in ij if p not in pseudo]
    ps = [p for p in ij if p in pseudo]
    pending = [tuple(p) for p in st.get("pending", []) if tuple(p) not in st["known_d"]]

    fig, axs = plt.subplots(1, 2, figsize=(11.5, 5), constrained_layout=True)
    kw = dict(cmap="viridis", vmin=0, vmax=1, extent=ext, aspect="auto")
    axs[0].imshow(load_truth(a.g), alpha=0.55, **kw)
    axs[0].set_title("Puntos KMC (%d medidos, %d pseudo) sobre el continuo g=%s" % (len(meas), len(ps), a.g))
    if len(ij) >= 4:
        rec, _ = kmc_planner.reconstruct(st)
        im = axs[1].imshow(rec, **kw)
    else:
        im = axs[1].imshow(np.full((st["res"],) * 2, np.nan), **kw)
    axs[1].set_title("Reconstrucción KMC con %d puntos (de %d)" % (len(ij), st["budget"]))
    for ax in axs:
        for pts, mk, s in ((meas, "o", 70), (ps, "^", 80)):
            if pts:
                p = np.array(pts)
                ax.scatter(le[p[:, 1]], lx[p[:, 0]], c=[st["known_d"][t] for t in pts], cmap="viridis",
                           vmin=0, vmax=1, marker=mk, s=s, edgecolors="w", linewidths=1.2)
        if pending:
            p = np.array(pending)
            ax.scatter(le[p[:, 1]], lx[p[:, 0]], marker="o", s=70, facecolors="none", edgecolors="k",
                       linewidths=1.2, linestyle="--")
        ax.set_xlabel(r"$\log(\ell)$")
        ax.set_ylabel(r"$\log(\Xi)$")
    for (i, j) in meas + ps:
        axs[0].annotate("%.2f" % st["known_d"][(i, j)], (le[j], lx[i]), xytext=(0, 9),
                        textcoords="offset points", ha="center", fontsize=7.5, color="k")
    fig.colorbar(im, ax=axs, label="SoC", shrink=0.9)
    if pending:
        axs[0].plot([], [], "o", mfc="none", mec="k", label="en curso / pedido")
    axs[0].plot([], [], "o", c="gray", label="KMC medido")
    if ps:
        axs[0].plot([], [], "^", c="gray", label="pseudo (cota + monotonía en ℓ)")
    axs[0].legend(loc="lower left", fontsize=8, framealpha=0.85)
    fig.savefig(out, dpi=140)
    print("->", out)


if __name__ == "__main__":
    main()
