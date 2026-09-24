#!/usr/bin/env python
# coding: utf-8
"""Punto 5 de la reunion del 17/9: reconstruir Rosenbrock y volver a mostrarlo como
superficie 3D (el camino inverso de la diapositiva "superficie -> vista de arriba -> corte").

Fila de arriba: superficies 3D (original y reconstrucciones). Fila de abajo: |error| en
vista de arriba, con los puntos observados cuando se conocen.

Reconstrucciones:
  - DIP con 1% / 5% / 10% observado (Bernoulli), de results/synthetic_targets_demo/rosenbrock/.
  - Interpolacion thin-plate (scipy) con 64 puntos en grilla 8x8, el presupuesto de autoexp.

Uso:  ./venv/bin/python -m analysis.rosenbrock_3d
Escribe results/rosenbrock_3d/rosenbrock_3d.png
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio

from autoexp import interp

IMG = os.environ.get("IMG", "data/restoration/synthetic/rosenbrock.png")
DIP_DIR = os.environ.get("DIP_DIR", "results/synthetic_targets_demo/rosenbrock")
OUT = os.environ.get("OUT", "results/rosenbrock_3d/rosenbrock_3d.png")


def load_dip(pct):
    r = np.load(os.path.join(DIP_DIR, "obs%dpct" % pct, "restored.npy"))[0]
    return r[1:-1, 1:-1] if r.shape[0] == 130 else r


def main():
    truth = np.asarray(Image.open(IMG).convert("L"), float) / 255.0
    H, W = truth.shape

    # 64 puntos en grilla 8x8 + TPS (mismo sampler que autoexp, sin oraculo: aca no hay presupuesto que auditar)
    ys = ((np.arange(8) + 0.5) * H / 8).astype(int)
    xs = ((np.arange(8) + 0.5) * W / 8).astype(int)
    ij = np.array([(y, x) for y in ys for x in xs])
    tps = interp.reconstruct(ij, truth[ij[:, 0], ij[:, 1]], (H, W), "rbf_tps")

    cols = [("Original", truth, None)]
    cols.append(("TPS, 64 pts (grilla 8x8)", tps, ij))
    for pct in (1, 5, 10):
        if os.path.isdir(os.path.join(DIP_DIR, "obs%dpct" % pct)):
            cols.append(("DIP, %d%% obs. (~%d pts)" % (pct, round(pct / 100 * H * W)), load_dip(pct), None))

    yy, xx = np.mgrid[0:H, 0:W]
    fig = plt.figure(figsize=(4 * len(cols), 8))
    for k, (name, rec, pts) in enumerate(cols):
        ax = fig.add_subplot(2, len(cols), k + 1, projection="3d")
        ax.plot_surface(xx, yy, np.clip(rec, 0, 1), cmap="viridis", vmin=0, vmax=1, rstride=2, cstride=2,
                        linewidth=0, antialiased=True)
        ax.set_zlim(0, 1)
        ax.view_init(elev=35, azim=-60)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        title = name if k == 0 else "%s\nPSNR %.1f dB" % (name, peak_signal_noise_ratio(truth, np.clip(rec, 0, 1), data_range=1.0))
        ax.set_title(title, fontsize=10)

        ax2 = fig.add_subplot(2, len(cols), len(cols) + k + 1)
        if k == 0:
            ax2.imshow(truth, cmap="viridis", vmin=0, vmax=1)
            ax2.set_title("vista de arriba", fontsize=9)
        else:
            ax2.imshow(np.abs(np.clip(rec, 0, 1) - truth), cmap="magma", vmin=0, vmax=0.2)
            ax2.set_title("|error| (escala 0-0.2)", fontsize=9)
        if pts is not None:
            ax2.scatter(pts[:, 1], pts[:, 0], s=6, c="c")
        ax2.set_xticks([]); ax2.set_yticks([])
    fig.suptitle("Rosenbrock: de la reconstruccion de vuelta a la superficie 3D", fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=110)
    print("->", OUT)


if __name__ == "__main__":
    main()
