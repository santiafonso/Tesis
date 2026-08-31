#!/usr/bin/env python
# coding: utf-8
"""Elige un subconjunto disperso de puntos (xi, el) sobre la misma grilla usada
por generate_phase_diagram.py (64x64, log in [-4,2]) para correrlos con KMC real
en el cluster. La fraccion de muestreo se fijo en base al barrido de
MASK_FRAC hecho sobre el mapa del modelo continuo: por debajo de ~2% el DIP
no reconstruye la frontera de fase; a partir de ~5% ya esta cerca del techo de
calidad (SSIM~0.96) con rendimientos decrecientes despues. Se usa 5% como
punto de partida costo/beneficio.

Al usar los mismos valores de xi/el que la grilla del continuo, la mascara real
resultante se puede comparar pixel a pixel contra `phase_diagram_soc.npy`
(el "gold standard" del modelo continuo) ademas de usarse como input real de
`restorationGRIS.py`.
"""
from __future__ import print_function

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEED = int(os.environ.get("SEED", "42"))
SAMPLE_FRAC = float(os.environ.get("SAMPLE_FRAC", "0.05"))

AXES_NPZ = "results_phase_diagram/phase_diagram_soc.npy".replace(
    "phase_diagram_soc.npy", "phase_diagram_soc_axes.npz"
)
REFERENCE_NPY = "results_phase_diagram/phase_diagram_soc.npy"

OUT_POINTS_TXT = "kmc_sample_points.txt"
OUT_MASK_NPY = "results_phase_diagram/kmc_sample_mask.npy"
OUT_PREVIEW_PNG = "results_phase_diagram/kmc_sample_preview.png"

axes = np.load(AXES_NPZ)
xi_values = axes["xi"]
ell_values = axes["ell"]
num_xi, num_ell = len(xi_values), len(ell_values)
total = num_xi * num_ell

rng = np.random.default_rng(SEED)
n_sample = round(SAMPLE_FRAC * total)
flat_idx = rng.choice(total, size=n_sample, replace=False)
xi_idx, ell_idx = np.unravel_index(flat_idx, (num_xi, num_ell))

mask = np.zeros((num_xi, num_ell), dtype=bool)
mask[xi_idx, ell_idx] = True
np.save(OUT_MASK_NPY, mask)

with open(OUT_POINTS_TXT, "w") as fh:
    for i, j in zip(xi_idx, ell_idx):
        fh.write("%.6f %.6f\n" % (xi_values[i], ell_values[j]))

print(
    "Elegidos %d/%d puntos (%.2f%%) -> %s (mascara en %s)"
    % (n_sample, total, 100.0 * n_sample / total, OUT_POINTS_TXT, OUT_MASK_NPY)
)

# --- Vista previa: puntos elegidos sobre el mapa de referencia del continuo ---
if os.path.exists(REFERENCE_NPY):
    soc_grid = np.load(REFERENCE_NPY)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        soc_grid,
        origin="lower",
        cmap="viridis",
        vmin=0,
        vmax=1,
        extent=[ell_values.min(), ell_values.max(), xi_values.min(), xi_values.max()],
        aspect="auto",
    )
    ax.scatter(ell_values[ell_idx], xi_values[xi_idx], s=10, c="red", label="puntos KMC elegidos")
    ax.set_xlabel(r"$\log(\ell)$")
    ax.set_ylabel(r"$\log(\Xi)$")
    ax.legend(loc="upper right")
    fig.colorbar(im, ax=ax, label=r"SoC$_{max}$")
    fig.tight_layout()
    fig.savefig(OUT_PREVIEW_PNG, dpi=150)
    plt.close(fig)
    print("Vista previa guardada en:", OUT_PREVIEW_PNG)
