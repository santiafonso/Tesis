#!/usr/bin/env python
# coding: utf-8
"""Resumen del barrido REG_NOISE_STD x familia x g (extra, sesion 15/9):
   3 g representativos (-4.0, -2.0, -0.5) x 3 familias (grid, uniform,
   frontier_mix) x 3 REG_NOISE_STD (0.01, 0.03, 0.08), mf=0.980, corrido en
   el cluster (slurm/dip_reg_noise_familias.slurm).

Uso:
    ./venv/bin/python -m analysis.reg_noise_familias_summary

Escribe:
    results/experimentos_menos_puntos/barrido_reg_noise_familias/summary.csv
    results/experimentos_menos_puntos/barrido_reg_noise_familias/psnr_by_family_g_reg.png
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GS = ["-4.0", "-2.0", "-0.5"]
FAMILIES = ["grid", "uniform", "frontier_mix"]
REGS = ["0.01", "0.03", "0.08"]
BASE = "results/experimentos_menos_puntos/barrido_reg_noise_familias"
OUT_DIR = BASE

COLORS = {"0.01": "#7f7f7f", "0.03": "#1f77b4", "0.08": "#d62728"}


def read_final(path):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        for row in csv.reader(f):
            if row and row[0] == "final":
                return float(row[1]), float(row[3])
    return None


def main():
    rows = []
    for g in GS:
        for fam in FAMILIES:
            for reg in REGS:
                p = os.path.join(BASE, "g%s" % g, "mf0.980", fam, "reg%s" % reg, "metrics.csv")
                r = read_final(p)
                if r is None:
                    print("!! falta", p)
                    continue
                rows.append({"g": g, "family": fam, "reg": reg, "psnr": r[0], "ssim": r[1]})

    csv_path = os.path.join(OUT_DIR, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["g", "family", "reg", "psnr", "ssim"])
        w.writeheader()
        w.writerows(rows)
    print("Escrito", csv_path)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    xw = 0.25
    for ax, g in zip(axs, GS):
        for fi, fam in enumerate(FAMILIES):
            for ri, reg in enumerate(REGS):
                sub = [r for r in rows if r["g"] == g and r["family"] == fam and r["reg"] == reg]
                if not sub:
                    continue
                x = fi + (ri - 1) * xw
                ax.bar(x, sub[0]["psnr"], width=xw, color=COLORS[reg],
                       label=("reg=%s" % reg) if fi == 0 else None)
        ax.set_xticks(range(len(FAMILIES)))
        ax.set_xticklabels(FAMILIES)
        ax.set_title("g = %s" % g)
        ax.grid(alpha=0.3, axis="y")
    axs[0].set_ylabel("PSNR final (dB)")
    axs[0].legend(title="REG_NOISE_STD", loc="upper left", fontsize=9)
    fig.suptitle(
        "PSNR segun REG_NOISE_STD, por familia y g (mf=0.980, ~328 pts)\n"
        "frontier_mix pasa de ser la peor familia a la mejor con reg=0.08"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    png_path = os.path.join(OUT_DIR, "psnr_by_family_g_reg.png")
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    print("Escrito", png_path)


if __name__ == "__main__":
    main()
