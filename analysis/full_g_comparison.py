#!/usr/bin/env python
# coding: utf-8
"""Tabla final (15/9): PSNR de frontier_mix + REG_NOISE_STD=0.08 en los 9 g,
a 328 puntos (2% obs., el metodo de referencia) vs. 82 puntos (0.5% obs.,
mismo metodo con menos presupuesto) vs. la busqueda binaria activa + 400
sinteticos en grilla (320 puntos, todos en la frontera) en los 3 g donde se
probo. Junta results/experimentos_menos_puntos/{barrido_reg_noise_familias,
fewpoints_3g,active_search,active_search_320}.

Uso:
    ./venv/bin/python -m analysis.full_g_comparison

Escribe:
    results/experimentos_menos_puntos/full_g_comparison.png
    results/experimentos_menos_puntos/full_g_comparison.csv
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

GS = ["-4.0", "-3.5", "-3.0", "-2.5", "-2.0", "-1.5", "-1.0", "-0.5", "0.0"]

# 328 pts, frontier_mix + reg=0.08 (los 3 de la corrida original + los 6 completados hoy)
PSNR_328 = {
    "-4.0": 41.91, "-3.5": 41.08, "-3.0": 42.00, "-2.5": 49.97,
    "-2.0": 42.90, "-1.5": 51.10, "-1.0": 51.90, "-0.5": 51.13, "0.0": 52.46,
}
# 82 pts, mismo metodo (frontier_mix + reg=0.08)
PSNR_82 = {"-4.0": 14.40, "-2.0": 26.26, "-0.5": 35.15}
# busqueda binaria activa (320 reales, solo frontera) + 400 sinteticos grilla
PSNR_ACTIVE_320 = {"-4.0": 29.52, "-2.0": 27.32, "-0.5": 31.95}


def main():
    out_csv = "results/experimentos_menos_puntos/full_g_comparison.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["g", "frontier_mix_reg0.08_328pts", "frontier_mix_reg0.08_82pts",
                    "active_search_320pts_grilla"])
        for g in GS:
            w.writerow([g, PSNR_328.get(g, ""), PSNR_82.get(g, ""), PSNR_ACTIVE_320.get(g, "")])
    print("Escrito", out_csv)

    fig, ax = plt.subplots(figsize=(11, 6))
    xs = list(range(len(GS)))
    ax.plot(xs, [PSNR_328[g] for g in GS], "o-", color="#1f77b4", linewidth=2, markersize=7,
            label="frontier_mix + reg=0.08 (328 pts, 2% obs.)")
    xs_82 = [i for i, g in enumerate(GS) if g in PSNR_82]
    ax.plot(xs_82, [PSNR_82[GS[i]] for i in xs_82], "s--", color="#ff7f0e", linewidth=2, markersize=8,
            label="mismo metodo, 82 pts (0.5% obs.)")
    xs_act = [i for i, g in enumerate(GS) if g in PSNR_ACTIVE_320]
    ax.plot(xs_act, [PSNR_ACTIVE_320[GS[i]] for i in xs_act], "^--", color="#d62728", linewidth=2, markersize=8,
            label="busqueda activa (320 pts, solo frontera) + sinteticos")
    ax.set_xticks(xs)
    ax.set_xticklabels(GS)
    ax.set_xlabel("g (interaccion de Frumkin)")
    ax.set_ylabel("PSNR final (dB)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_title(
        "PSNR vs. g, para 3 presupuestos de puntos (todas con REG_NOISE_STD=0.08)\n"
        "el metodo de sinteticos solo le gana al simple cuando el frente es abrupto (g muy negativo)"
    )
    fig.tight_layout()
    out_png = "results/experimentos_menos_puntos/full_g_comparison.png"
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print("Escrito", out_png)


if __name__ == "__main__":
    main()
