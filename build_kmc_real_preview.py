#!/usr/bin/env python
# coding: utf-8
"""Arma un preview con TODOS los puntos que el job kmc_sparse_1167939 lanzo
(52, listados en kmc_launched_points.txt, extraidos del log del cluster antes
de que el job hiciera timeout), superpuestos sobre el mapa de referencia
continuo. Cada punto se clasifica segun lo que efectivamente se encontro en
results_phase_diagram/kmc_real/:
  - con datos reales -> circulo relleno, coloreado por su SoC_fin real
  - lanzado pero solo con el header del .dat (mato al proceso antes del
    primer paso KMC) -> equis gris
  - lanzado pero sin ningun .dat en disco (mato al proceso durante el setup,
    antes de abrir el archivo) -> triangulo hueco
"""
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REAL_DIR = "results_phase_diagram/kmc_real"
REFERENCE_NPY = "results_phase_diagram/phase_diagram_soc.npy"
AXES_NPZ = "results_phase_diagram/phase_diagram_soc_axes.npz"
LAUNCHED_TXT = "kmc_launched_points.txt"
OUT_PNG = "results_phase_diagram/kmc_real_preview.png"

axes = np.load(AXES_NPZ)
xi_values, ell_values = axes["xi"], axes["ell"]
soc_grid = np.load(REFERENCE_NPY)

pat = re.compile(r"xi(m?)(\d+p\d+)-el(m?)(\d+p\d+)")


def parse_signed(fname):
    m = pat.search(fname)
    xi_neg, xi_tok, el_neg, el_tok = m.groups()
    xi = float(xi_tok.replace("p", "."))
    el = float(el_tok.replace("p", "."))
    if xi_neg:
        xi = -xi
    if el_neg:
        el = -el
    return xi, el


# --- indexar lo que hay en disco por (xi, el) redondeado ---
disk = {}
for path in sorted(glob.glob(os.path.join(REAL_DIR, "*.dat"))):
    fname = os.path.basename(path)
    if not pat.search(fname):
        continue
    xi, el = parse_signed(fname)
    with open(path) as fh:
        lines = [l.strip() for l in fh if l.strip()]
    if len(lines) >= 2:
        soc_final = float(lines[-1].split()[0])
        disk[(round(xi, 4), round(el, 4))] = ("data", soc_final, fname)
    else:
        disk[(round(xi, 4), round(el, 4))] = ("stub", None, fname)

# --- clasificar los 52 puntos lanzados ---
with open(LAUNCHED_TXT) as fh:
    launched = [tuple(map(float, l.split())) for l in fh if l.strip()]

with_data, stub_only, no_file = [], [], []
for xi, el in launched:
    key = (round(xi, 4), round(el, 4))
    status = disk.get(key)
    if status is None:
        no_file.append((xi, el))
    elif status[0] == "data":
        with_data.append((xi, el, status[1]))
    else:
        stub_only.append((xi, el))

print("Lanzados: %d | con datos: %d | solo header: %d | sin archivo: %d" % (
    len(launched), len(with_data), len(stub_only), len(no_file)))

fig, ax = plt.subplots(figsize=(7.2, 5.8))
im = ax.imshow(
    soc_grid,
    origin="lower",
    cmap="viridis",
    vmin=0,
    vmax=1,
    extent=[ell_values.min(), ell_values.max(), xi_values.min(), xi_values.max()],
    aspect="auto",
)

if with_data:
    xs = [p[1] for p in with_data]
    ys = [p[0] for p in with_data]
    cs = [p[2] for p in with_data]
    ax.scatter(xs, ys, c=cs, cmap="viridis", vmin=0, vmax=1, s=70,
               edgecolors="red", linewidths=1.4, zorder=3,
               label="con datos reales (%d)" % len(with_data))

if stub_only:
    xs = [p[1] for p in stub_only]
    ys = [p[0] for p in stub_only]
    ax.scatter(xs, ys, marker="x", c="dimgray", s=60, linewidths=2, zorder=3,
               label="lanzado, sin datos (solo header) (%d)" % len(stub_only))

if no_file:
    xs = [p[1] for p in no_file]
    ys = [p[0] for p in no_file]
    ax.scatter(xs, ys, marker="^", facecolors="none", edgecolors="white",
               s=60, linewidths=1.6, zorder=3,
               label="lanzado, sin archivo .dat (%d)" % len(no_file))

ax.set_xlabel(r"$\log(\ell)$")
ax.set_ylabel(r"$\log(\Xi)$")
ax.set_title("Puntos lanzados en el sweep KMC disperso (job 1167939, %d/205)" % len(launched))
ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
fig.colorbar(im, ax=ax, label=r"SoC$_{fin}$ (fondo: continuo; puntos con datos: SoC KMC real)")
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=150)
print("Guardado:", OUT_PNG)

print()
print("Puntos con datos reales:")
for xi, el, soc in sorted(with_data, key=lambda p: p[2]):
    print("  log(Xi)=%8.4f log(l)=%8.4f SoC_fin=%.4f" % (xi, el, soc))
