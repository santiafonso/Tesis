"""Reconstruccion KMC al lado del mapa del modelo del continuo con el mismo g, y la diferencia.

    python -m autoexp.kmc_vs_continuo --state autoexp/kmc/g3/kmc.json
-> autoexp/kmc/g3/vs_continuo.png

Usa dip/final.npy (TPS + DIP) si existe, si no mapa_kmc.npy (TPS). El continuo sale de
results/phase_diagram_g/g<g>/soc.npy; si no hay para ese g, se muestran los dos g vecinos.
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from autoexp import kmc_planner
from autoexp.oracle import load_truth

GS = (-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 1.0, 2.0, 3.0, 4.0)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def continuo(g):
    d = os.path.join(REPO, "results", "phase_diagram_g", "g%.1f" % g)
    return load_truth("%.1f" % g) if os.path.isfile(os.path.join(d, "sim_128.png")) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--tps", action="store_true", help="usar la TPS aunque haya dip/final.npy")
    a = ap.parse_args()
    st = kmc_planner.load(a.state)
    here = os.path.dirname(a.state)
    g = float(st.get("g", -4.0))
    lx, le = kmc_planner.axes(st)
    ext = [le[0], le[-1], lx[-1], lx[0]]
    fin = os.path.join(here, "dip", "final.npy")
    rec, nombre = (np.load(fin), "TPS + DIP") if os.path.isfile(fin) and not a.tps else (kmc_planner.reconstruct(st)[0], "TPS")
    (ij, v), (ijp, _) = kmc_planner.split(st)
    c = continuo(g)
    if c is not None:
        refs = [(c, "continuo g = %+g" % g)]
    else:
        vec = sorted((x for x in GS if continuo(x) is not None), key=lambda x: abs(x - g))[:2]
        refs = [(continuo(x), "continuo g = %+g (no hay %+g)" % (x, g)) for x in sorted(vec)]
    paneles = [(rec, "KMC g = %+g: %s, %d puntos" % (g, nombre, len(st["known_d"])))] + refs
    if c is not None:
        paneles.append((rec - c, "KMC − continuo"))
    fig, axs = plt.subplots(1, len(paneles), figsize=(5.6 * len(paneles), 5), squeeze=False)
    for ax, (m, t) in zip(axs[0], paneles):
        dif = t.startswith("KMC −")
        im = ax.imshow(m, cmap="RdBu_r" if dif else "viridis", vmin=-0.5 if dif else 0, vmax=0.5 if dif else 1,
                       extent=ext, aspect="auto")
        if ax is axs[0][0] or dif:
            ax.scatter(le[ij[:, 1]], lx[ij[:, 0]], c="w", s=12, edgecolors="k", linewidths=0.3, label="medidos")
            if len(ijp):
                ax.scatter(le[ijp[:, 1]], lx[ijp[:, 0]], facecolors="none", edgecolors="r", s=45,
                           label="pseudo (cota)")
        ax.set_title(t)
        ax.set_xlabel(r"$\log \ell$")
        ax.set_ylabel(r"$\log \Xi$")
        fig.colorbar(im, ax=ax, fraction=0.046)
    axs[0][0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = os.path.join(here, "vs_continuo.png")
    fig.savefig(out, dpi=120)
    print("->", out)


if __name__ == "__main__":
    main()
