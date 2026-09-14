#!/usr/bin/env python
# coding: utf-8
"""Punto 3 de la reunion del 9/9: comparar las 5 distribuciones de muestreo
(uniform, grid, frontier, frontier_mix, spread) en el regimen de g negativo.

Lee results/dip_gsweep_frontier/g<val>/mf<frac>/<family>/metrics.csv (225
corridas: 9 g x 5 mf x 5 familias) y arma:
    results/gsweep_5family_summary/summary.csv
    results/gsweep_5family_summary/psnr_vs_pct_by_g.png   (grilla 3x3, una por g)

Pensado para entrar directo a una diapositiva (punto 5: destacar g negativo).

Uso:
    ./venv/bin/python -m analysis.gsweep_5family_summary
"""
from __future__ import print_function

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

GS = [-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0]
MFS = ["0.900", "0.950", "0.980", "0.990", "0.995"]
FAMILIES = ["uniform", "grid", "spread", "frontier", "frontier_mix"]
RESULTS_DIR = "results/dip_gsweep_frontier"
OUT_DIR = "results/gsweep_5family_summary"

COLORS = {
    "uniform": "#7f7f7f",
    "grid": "#1f77b4",
    "spread": "#2ca02c",
    "frontier": "#d62728",
    "frontier_mix": "#ff7f0e",
}
MARKERS = {
    "uniform": "o", "grid": "s", "spread": "v", "frontier": "D", "frontier_mix": "^",
}

os.makedirs(OUT_DIR, exist_ok=True)


def read_final_metrics(metrics_csv):
    if not os.path.exists(metrics_csv):
        return None
    rows = {}
    with open(metrics_csv) as f:
        for row in csv.reader(f):
            if not row or row[0] in ("iter", ""):
                continue
            rows[row[0]] = row
    final = rows.get("final")
    mae = rows.get("mae")
    fallbacks = rows.get("fallbacks")
    if final is None:
        return None
    return {
        "psnr_final": float(final[1]) if final[1] else float("nan"),
        "ssim_final": float(final[3]) if final[3] else float("nan"),
        "mae_final": float(mae[1]) if mae and mae[1] else float("nan"),
        "fallbacks": int(float(fallbacks[1])) if fallbacks and fallbacks[1] else -1,
    }


def main():
    records = []
    for g in GS:
        for mf in MFS:
            pct_obs = 100.0 * (1.0 - float(mf))
            for fam in FAMILIES:
                metrics_path = os.path.join(RESULTS_DIR, "g%s" % g, "mf%s" % mf, fam, "metrics.csv")
                m = read_final_metrics(metrics_path)
                if m is None:
                    print("!! falta", metrics_path)
                    continue
                rec = {"g": g, "mf": mf, "pct_obs": pct_obs, "family": fam}
                rec.update(m)
                records.append(rec)

    if not records:
        print("Sin registros -- corré primero slurm/dip_gsweep_frontier.slurm y sincronizá results/.")
        return

    csv_path = os.path.join(OUT_DIR, "summary.csv")
    fieldnames = ["g", "mf", "pct_obs", "family", "psnr_final", "ssim_final", "mae_final", "fallbacks"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(records)
    print("Escrito", csv_path, "(%d filas)" % len(records))

    # --- grilla 3x3 (una por g), PSNR vs. % observado, 5 lineas -----------
    fig, axs = plt.subplots(3, 3, figsize=(16, 13), sharex=True, sharey=False)
    for ax, g in zip(axs.flat, GS):
        for fam in FAMILIES:
            sub = sorted([r for r in records if r["g"] == g and r["family"] == fam],
                         key=lambda r: r["pct_obs"])
            xs = [r["pct_obs"] for r in sub]
            ys = [r["psnr_final"] for r in sub]
            ax.plot(xs, ys, marker=MARKERS[fam], color=COLORS[fam], label=fam,
                     linewidth=1.8, markersize=5)
        ax.set_title("g = %s" % g, fontsize=11)
        ax.set_xscale("log")
        ax.grid(alpha=0.3)
    for ax in axs[-1, :]:
        ax.set_xlabel("% observado")
    for ax in axs[:, 0]:
        ax.set_ylabel("PSNR final (dB)")
    axs[0, 0].legend(loc="best", fontsize=9)
    fig.suptitle(
        "Punto 3 (9/9): fidelidad vs. %% observado, 5 distribuciones, g negativo\n"
        "(punto 5: g negativo es el caso real -> foco de la entrega)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    png_path = os.path.join(OUT_DIR, "psnr_vs_pct_by_g.png")
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    print("Escrito", png_path)


if __name__ == "__main__":
    main()
