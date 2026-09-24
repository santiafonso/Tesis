#!/usr/bin/env python
# coding: utf-8
"""Grilla Original vs. Reconstruida (el mejor caso encontrado: frontier_mix
+ REG_NOISE_STD=0.08, 328 pts) para los 3 g representativos del barrido de
la sesion 15/9. Usa el mismo panel `comparison_annotated.png` que ya escribe
dip.restoration (Original | Enmascarada | Reconstruida | error), uno por g.

Uso:
    ./venv/bin/python -m analysis.best_per_g_comparison

Escribe:
    results/experimentos_menos_puntos/barrido_reg_noise_familias/best_per_g.png
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

GS = ["-4.0", "-3.5", "-3.0", "-2.5", "-2.0", "-1.5", "-1.0", "-0.5", "0.0"]
BASE = "results/experimentos_menos_puntos/barrido_reg_noise_familias"
OUT = os.path.join(BASE, "best_per_g.png")


def main():
    fig, axs = plt.subplots(len(GS), 1, figsize=(11, 3.2 * len(GS)))
    for ax, g in zip(axs, GS):
        img_path = os.path.join(BASE, "g%s" % g, "mf0.980", "frontier_mix", "reg0.08", "comparison_annotated.png")
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title("g = %s  --  frontier_mix + REG_NOISE_STD=0.08 (328 pts, 2%% obs.)" % g,
                     fontsize=12, loc="left")
    fig.suptitle("Original | Enmascarada | Reconstruida | error -- mejor caso por g", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT, dpi=130)
    plt.close(fig)
    print("Escrito", OUT)


if __name__ == "__main__":
    main()
