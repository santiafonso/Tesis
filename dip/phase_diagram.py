#!/usr/bin/env python
# coding: utf-8
"""Genera el diagrama de fases SoC_fin(log(Xi), log(l)) con el modelo continuo
(single-particle model + Butler-Volmer, isoterma de Langmuir), usando el paquete
`galpynostatic` -- la misma herramienta y version (0.5.13) que usa
entropy-27-00663.pdf para construir su Figura 2b/11f -- en vez de un barrido KMC
real. Es rapido y corre local (sin cluster): sirve como imagen "original" densa
para probar el pipeline de reconstruccion DIP (dip.inpaint) con muestreo
disperso, antes de gastar tiempo de cluster en puntos KMC reales.

Se ejecuta como modulo desde la raiz del repo:
    ./venv/bin/python -m dip.phase_diagram

Salidas:
  - data/restoration/phase_diagram_sim.png : escala de grises, valores crudos
    de SoC_fin normalizados [0,1] -- este es el input que consume DIP.
  - results_phase_diagram/phase_diagram_reference.png : mismo mapa coloreado
    (viridis + ejes + colorbar), solo para comparacion visual con el paper.
  - results_phase_diagram/phase_diagram_soc.npy(+_axes.npz) : grilla cruda y
    los valores de log(Xi)/log(l) de cada fila/columna.
"""
from __future__ import print_function

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import galpynostatic.simulation as gpsim

# --- Config (variables de entorno, mismo patron que dip.inpaint) ---
NUM_XI = int(os.environ.get("NUM_XI", "64"))
NUM_ELL = int(os.environ.get("NUM_ELL", "64"))
GRID_SIZE = int(os.environ.get("GRID_SIZE", "1000"))
TIME_STEPS = int(os.environ.get("TIME_STEPS", "100000"))

LOGXI_LOW, LOGXI_HIGH = -4.0, 2.0
LOGELL_LOW, LOGELL_HIGH = -4.0, 2.0
VCUT = -0.15  # phi_cut, igual que en el paper
G = 0.0  # interaccion nula -> isoterma de Langmuir, igual que en Fig. 2b del paper

OUT_PNG = os.environ.get("OUT_PNG", "./data/restoration/phase_diagram_sim.png")
OUT_REF_PNG = os.environ.get(
    "OUT_REF_PNG", "./results_phase_diagram/phase_diagram_reference.png"
)
OUT_NPY = os.environ.get("OUT_NPY", "./results_phase_diagram/phase_diagram_soc.npy")

os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
os.makedirs(os.path.dirname(OUT_REF_PNG), exist_ok=True)
os.makedirs(os.path.dirname(OUT_NPY), exist_ok=True)

print(
    "Calculando mapa SoC_fin(log Xi, log l) con galpynostatic: %dx%d puntos "
    "(grid_size=%d, time_steps=%d)..." % (NUM_XI, NUM_ELL, GRID_SIZE, TIME_STEPS)
)
gm = gpsim.GalvanostaticMap(
    vcut=VCUT,
    g=G,
    logxi_lle=LOGXI_HIGH,
    logxi_ule=LOGXI_LOW,
    num_xi=NUM_XI,
    logell_lle=LOGELL_HIGH,
    logell_ule=LOGELL_LOW,
    num_ell=NUM_ELL,
    grid_size=GRID_SIZE,
    time_steps=TIME_STEPS,
    nthreads=-1,
)
gm.run()
df = gm.map_dataframe

# --- Pivot a grilla 2D: filas = xi ascendente, columnas = ell ascendente ---
pivot = df.pivot(index="xi", columns="ell", values="SOC").sort_index(axis=0).sort_index(axis=1)
xi_values = pivot.index.to_numpy()
ell_values = pivot.columns.to_numpy()
soc_grid = pivot.to_numpy(dtype=np.float32)

np.save(OUT_NPY, soc_grid)
np.savez(OUT_NPY.replace(".npy", "_axes.npz"), xi=xi_values, ell=ell_values)
print("Grilla cruda guardada en:", OUT_NPY, soc_grid.shape)

# --- PNG en escala de grises (input DIP): fila 0 = xi mas alto, igual que la ---
# --- figura con origin='lower' de abajo hacia arriba.                      ---
soc_img = np.flipud(np.clip(soc_grid, 0.0, 1.0))
soc_img = (soc_img * 255).astype(np.uint8)
Image.fromarray(soc_img, mode="L").save(OUT_PNG)
print("Imagen (input DIP) guardada en:", OUT_PNG, soc_img.shape)

# --- Render coloreado tipo paper (Fig. 2b/11f): viridis + colorbar, solo ---
# --- para referencia visual, no se usa como input de DIP.                ---
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(
    soc_grid,
    origin="lower",
    cmap="viridis",
    vmin=0,
    vmax=1,
    extent=[LOGELL_LOW, LOGELL_HIGH, LOGXI_LOW, LOGXI_HIGH],
    aspect="auto",
)
ax.set_xlabel(r"$\log(\ell)$")
ax.set_ylabel(r"$\log(\Xi)$")
fig.colorbar(im, ax=ax, label=r"SoC$_{max}$")
fig.tight_layout()
fig.savefig(OUT_REF_PNG, dpi=150)
plt.close(fig)
print("Referencia visual guardada en:", OUT_REF_PNG)
print("Listo.")
