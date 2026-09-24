#!/usr/bin/env python
# coding: utf-8
"""Version simplificada de la slide del punto 2 (9/9): un solo scatter,
PSNR final vs. cuantos puntos observados caen sobre la frontera, marcando
el caso minimo que ya es confiable.

No regenera nada: lee results/frontier_point_count/summary.csv (ya escrito
por analysis/frontier_point_count.py).

Uso:
    ./venv/bin/python -m analysis.frontier_point_count_simple

Escribe:
    results/frontier_point_count/psnr_vs_frontier_points_simple.png
"""
from __future__ import print_function

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN_CSV = "results/frontier_point_count/summary.csv"
OUT_PNG = "results/frontier_point_count/psnr_vs_frontier_points_simple.png"
PSNR_OK = 35.0  # umbral "reconstruccion confiable" usado en la presentacion3
GOOD_FAMILIES = ("grid", "uniform")  # reparten parejo -- son las que importan


def main():
    rows = list(csv.DictReader(open(IN_CSV)))
    for r in rows:
        r["n_on_edge"] = int(r["n_on_edge"])
        r["n_obs"] = int(r["n_obs"])
        r["psnr_final"] = float(r["psnr_final"])

    good = [r for r in rows if r["family"] in GOOD_FAMILIES]
    rest = [r for r in rows if r["family"] not in GOOD_FAMILIES]

    # el caso minimo confiable: menor n_on_edge entre las corridas >= PSNR_OK
    confiables = [r for r in good if r["psnr_final"] >= PSNR_OK]
    best = min(confiables, key=lambda r: r["n_on_edge"])

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter([r["n_on_edge"] for r in rest], [r["psnr_final"] for r in rest],
               c="#c9c9c9", s=35, alpha=0.6, label="frontier / frontier_mix / spread "
               "(concentran puntos en la frontera)", zorder=1)
    ax.scatter([r["n_on_edge"] for r in good], [r["psnr_final"] for r in good],
               c="#1f77b4", s=55, edgecolors="k", linewidths=0.4,
               label="grid / uniform (reparten parejo)", zorder=2)

    ax.axhline(PSNR_OK, color="#d62728", linestyle="--", linewidth=1.3, zorder=0)
    ax.text(5, PSNR_OK + 1.0,
            "PSNR = %.0f dB (umbral “confiable”)" % PSNR_OK,
            color="#d62728", fontsize=10)

    ax.scatter([best["n_on_edge"]], [best["psnr_final"]], marker="*", s=420,
               c="#ffcc00", edgecolors="k", linewidths=1.0, zorder=3)
    ax.annotate(
        "minimo confiable:\n%d puntos en la frontera\n(%d puntos en total, g=%s, %s)\n%.1f dB"
        % (best["n_on_edge"], best["n_obs"], best["g"], best["family"], best["psnr_final"]),
        xy=(best["n_on_edge"], best["psnr_final"]),
        xytext=(280, 20),
        fontsize=10, ha="left",
        arrowprops=dict(arrowstyle="->", color="k", linewidth=1.0),
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#999999"),
    )

    ax.set_xlabel("# puntos observados que caen sobre la frontera")
    ax.set_ylabel("PSNR final (dB)")
    ax.set_title("Cuantos puntos en la frontera hacen falta para una reconstruccion confiable")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130)
    plt.close(fig)
    print("Escrito", OUT_PNG)


if __name__ == "__main__":
    main()
