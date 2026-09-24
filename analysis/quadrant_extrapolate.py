#!/usr/bin/env python
# coding: utf-8
"""Concentrar los puntos en un cuadrante chico, reconstruirlo bien (alta
densidad local), y EXTRAPOLAR la curva de frontera al resto de la imagen en
vez de gastar puntos ahi -- la idea: como la forma de la frontera es
consistente entre mapas, con un pedazo bien reconstruido alcanza para saber
para donde sigue.

Ajusta una recta a cada tramo de la curva detectada DENTRO de la ventana
(arriba del codo = tramo casi vertical, abajo = tramo diagonal) y extiende
esas rectas al resto de la imagen; redibuja con rebuild_clean_line.

Uso:
    ./venv/bin/python -m analysis.quadrant_extrapolate <run_dir> <window_dir>
"""
from __future__ import print_function

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dip import metrics as dip_metrics  # noqa: E402
from analysis.correct_frontier_line import (  # noqa: E402
    crop_pad, load_gray, load_original, row_crossings, smooth_fill, rebuild_clean_line,
)
from analysis.window_square import find_elbow  # noqa: E402


def main():
    run_dir = sys.argv[1]
    window_dir = sys.argv[2]
    base = os.path.basename(window_dir)
    prep_dir = os.path.join(run_dir, base[:base.index("_dip")])
    meta = json.load(open(os.path.join(prep_dir, "meta.json")))
    size, r0, c0 = meta["size"], meta["r0"], meta["c0"]

    win_rec = crop_pad(load_gray(os.path.join(window_dir, "restored.png")))
    original = load_original(run_dir)
    H, W = original.shape

    xs_local = row_crossings(win_rec)
    xs_local_s = smooth_fill(xs_local, 5)
    # el codo ya se ubico (con mas contexto, 128 filas) al armar la ventana
    # (prepare() en window_square.py, guardado en meta.json) -- re-detectarlo
    # en el recorte de 64 filas es mucho menos confiable (poco contexto para
    # estimar curvatura), asi que se reusa ese valor en vez de recalcular.
    elbow_local = int(round(meta["cy"] - r0))
    elbow_local = int(np.clip(elbow_local, 0, size - 1))
    print("codo local (reusado de la ubicacion de la ventana) en fila=%d" % elbow_local)

    rows_local = np.arange(size)
    good = ~np.isnan(xs_local_s)

    def fit_segment(lo, hi):
        sel = good & (rows_local >= lo) & (rows_local < hi)
        if sel.sum() < 2:
            return None
        a, b = np.polyfit(rows_local[sel], xs_local_s[sel], 1)
        return a, b

    up = fit_segment(0, elbow_local + 1)
    lo = fit_segment(elbow_local, size)
    print("tramo superior (a,b)=", up, " tramo inferior (a,b)=", lo)

    xs_full = np.full(H, np.nan)
    for row in range(H):
        local_row = row - r0
        if r0 <= row < r0 + size and not np.isnan(xs_local_s[local_row]):
            xs_full[row] = c0 + xs_local_s[local_row]
        elif local_row < elbow_local and up is not None:
            xs_full[row] = c0 + up[0] * local_row + up[1]
        elif lo is not None:
            xs_full[row] = c0 + lo[0] * local_row + lo[1]
    # OJO: no recortar xs_full a [0, W) aca -- si la curva extrapolada ya
    # salio del cuadro (x negativo o > W), eso es la señal correcta de "esta
    # fila ya es meseta pura"; rebuild_clean_line la satura sola via su
    # propio clip interno. Recortar el CENTRO antes deja la mitad de la
    # banda de transicion "flotando" justo en el borde (gris en vez de
    # negro/blanco puro).

    high = np.quantile(win_rec, 0.95)
    low = np.quantile(win_rec, 0.05)
    # ancho de transicion: medido dentro de la ventana (10%-90%), en vez de
    # inventado
    lvl_hi = high - 0.1 * (high - low)
    lvl_lo = low + 0.1 * (high - low)
    xs_hi = row_crossings(win_rec, level=lvl_hi)
    xs_lo = row_crossings(win_rec, level=lvl_lo)
    widths = (xs_lo - xs_hi)[~np.isnan(xs_lo) & ~np.isnan(xs_hi)]
    width = float(np.median(widths)) if len(widths) else 10.0
    print("meseta alta=%.3f baja=%.3f ancho estimado=%.1f" % (high, low, width))

    rebuilt = rebuild_clean_line(np.zeros((H, W)), xs_full, high, low, width)
    rebuilt = np.clip(rebuilt, 0, 1)

    def ps(a, b):
        return dip_metrics.psnr(a[None], b[None]), dip_metrics.ssim(a[None], b[None], 1)

    p1, s1 = ps(original, rebuilt)
    print("Extrapolado:  PSNR %.2f  SSIM %.4f" % (p1, s1))

    out_png = os.path.join(run_dir, "quadrant_extrapolated.png")
    Image.fromarray((rebuilt * 255).astype(np.uint8), mode="L").save(out_png)

    fig, axs = plt.subplots(1, 3, figsize=(10.5, 4))
    axs[0].imshow(original, cmap="gray"); axs[0].set_title("Original")
    axs[0].add_patch(plt.Rectangle((c0, r0), size, size, fill=False, edgecolor="red", linewidth=1.5))
    axs[1].imshow(win_rec, cmap="gray")
    axs[1].set_title("Ventana sola (%dx%d, alta densidad)" % (size, size))
    axs[2].imshow(rebuilt, cmap="gray")
    y = np.arange(H)
    axs[2].plot(xs_full, y, color="lime", linewidth=0.8, alpha=0.6)
    axs[2].set_title("Extrapolada (PSNR %.1f, SSIM %.3f)" % (p1, s1))
    for ax in (axs[0], axs[2]):
        ax.set_xticks([]); ax.set_yticks([])
    axs[1].set_xticks([]); axs[1].set_yticks([])
    fig.suptitle(run_dir)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    cmp_png = os.path.join(run_dir, "quadrant_extrapolate_comparison.png")
    fig.savefig(cmp_png, dpi=130)
    plt.close(fig)
    print("Escrito", out_png, "y", cmp_png)


if __name__ == "__main__":
    main()
