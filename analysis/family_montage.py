#!/usr/bin/env python
# coding: utf-8
"""Monta las 5 reconstrucciones (restored.png) de un (g, mf) lado a lado,
con PSNR/SSIM de metrics.csv como titulo, y superpone el contorno de la
frontera REAL (calculada del soc.npy denso del modelo del continuo, mismo
metodo que dip.frontier_mask: |grad SoC| suavizado, contorno al percentil 90)
sobre cada panel -- punto 4 de la reunion del 9/9: ver donde el DIP se
desvia de la frontera teorica conocida.

Uso:
    G=-4.0 MF=0.980 ./venv/bin/python -m analysis.family_montage
"""
from __future__ import print_function

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from PIL import Image

G = os.environ.get("G", "-4.0")
MF = os.environ.get("MF", "0.980")
FAMILIES = ["uniform", "grid", "spread", "frontier", "frontier_mix"]
RESULTS_DIR = "results/dip_gsweep_frontier"
PHASE_G_DIR = "results/phase_diagram_g"
OUT_DIR = "results/gsweep_5family_summary"
SMOOTH_SIGMA = 1.0
EDGE_QUANTILE = 0.90

os.makedirs(OUT_DIR, exist_ok=True)


def read_final(metrics_csv):
    rows = {}
    with open(metrics_csv) as f:
        for row in csv.reader(f):
            if not row or row[0] in ("iter", ""):
                continue
            rows[row[0]] = row
    final = rows.get("final")
    return float(final[1]), float(final[3])


def true_frontier_curve(g, shape):
    """Curva de frontera real (una sola linea), del soc.npy denso (o el PNG
    como proxy si falta). |grad SoC| da una CRESTA (maxima en el borde, cae
    a los dos lados) -- contornear esa cresta a un nivel por debajo del pico
    (p.ej. el percentil 90 usado antes) cruza DOS VECES por fila, una a cada
    lado, y dibuja una falsa doble linea/forma de Z en vez de la frontera.
    Ahora se traza el PICO de la cresta fila por fila (argmax, con
    refinamiento sub-pixel por interpolacion parabolica) -- una sola curva,
    sin el artefacto."""
    soc_path = os.path.join(PHASE_G_DIR, "g%s" % g, "soc.npy")
    if os.path.exists(soc_path):
        soc = np.load(soc_path).astype(np.float32)
        soc = np.flipud(soc)
    else:
        png_path = os.path.join(PHASE_G_DIR, "g%s" % g, "sim_128.png")
        if not os.path.exists(png_path):
            return None, None
        soc = np.asarray(Image.open(png_path).convert("L"), dtype=np.float32) / 255.0
    gy, gx = np.gradient(ndi.gaussian_filter(soc, SMOOTH_SIGMA))
    gmag = np.hypot(gx, gy)
    gmag = ndi.gaussian_filter(gmag, SMOOTH_SIGMA)
    edge = gmag / max(gmag.max(), 1e-12)

    H, W = edge.shape
    xs = np.full(H, np.nan)
    for y in range(H):
        row = edge[y]
        k = int(np.argmax(row))
        if row[k] < 0.05:  # sin borde real en esta fila (meseta plana)
            continue
        if 0 < k < W - 1:
            v0, v1, v2 = row[k - 1], row[k], row[k + 1]
            denom = v0 - 2 * v1 + v2
            delta = 0.5 * (v0 - v2) / denom if denom != 0 else 0.0
            delta = float(np.clip(delta, -1, 1))
        else:
            delta = 0.0
        xs[y] = k + delta
    ys = np.arange(H, dtype=np.float32)
    if (H, W) != shape:
        ys = ys * (shape[0] / H)
        xs = xs * (shape[1] / W)
    return xs, ys


fig, axs = plt.subplots(1, len(FAMILIES) + 1, figsize=(3.2 * (len(FAMILIES) + 1), 3.6))

base_dir = os.path.join(RESULTS_DIR, "g%s" % G, "mf%s" % MF)
orig_shown = False
curve_xs, curve_ys, curve_shape = None, None, None
panels = []  # (ax, is_original)

for i, fam in enumerate(FAMILIES):
    run_dir = os.path.join(base_dir, fam)
    restored_path = os.path.join(run_dir, "restored.png")
    metrics_path = os.path.join(run_dir, "metrics.csv")
    ax = axs[i + 1]
    if not os.path.exists(restored_path):
        ax.set_title("%s\n(falta)" % fam)
        ax.axis("off")
        continue
    img = Image.open(restored_path)
    if not orig_shown:
        fc_path = os.path.join(run_dir, "final_comparison.png")
        if os.path.exists(fc_path):
            fc = Image.open(fc_path)
            w = fc.width // 4  # panel crudo: original | enmascarada | reconstruida | error
            orig_crop = fc.crop((0, 0, w, fc.height))
            axs[0].imshow(orig_crop, cmap="gray")
            axs[0].set_title("Original")
            axs[0].axis("off")
            panels.append((axs[0], orig_crop.size[::-1]))
        curve_shape = img.size[::-1]
        curve_xs, curve_ys = true_frontier_curve(G, curve_shape)
        orig_shown = True
    psnr, ssim = read_final(metrics_path)
    ax.imshow(img, cmap="gray")
    ax.set_title("%s\nPSNR=%.1f SSIM=%.3f" % (fam, psnr, ssim), fontsize=10)
    ax.axis("off")
    panels.append((ax, img.size[::-1]))

if curve_xs is not None:
    for ax, shape in panels:
        xs, ys = curve_xs, curve_ys
        if shape != curve_shape:
            xs = xs * (shape[1] / curve_shape[1])
            ys = ys * (shape[0] / curve_shape[0])
        ax.plot(xs, ys, color="lime", linewidth=1.2, alpha=0.85)

fig.suptitle(
    "g=%s  mf=%s  (%.1f%% observado)  --  linea verde = frontera real "
    "(soc.npy, cresta de |grad|)"
    % (G, MF, 100 * (1 - float(MF))),
    fontsize=12,
)
fig.tight_layout(rect=[0, 0, 1, 0.92])
out_path = os.path.join(OUT_DIR, "montage_g%s_mf%s.png" % (G, MF))
fig.savefig(out_path, dpi=130)
plt.close(fig)
print("Escrito", out_path)
