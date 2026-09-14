#!/usr/bin/env python
# coding: utf-8
"""Ventana centralizada en la frontera, idea del profe: en vez de correr DIP
sobre la imagen 128x128 completa, se "endereza" la imagen (se corre cada fila
para que la curva de frontera quede vertical, usando la MISMA correccion de
analysis/correct_frontier_line.py, sin ver la imagen real) y se recorta una
franja angosta (64 px, el minimo que acepta dip.restoration por el
crop_image(...,64)) centrada en esa curva. DIP corre solo sobre esa franja
--con los puntos que caen adentro-- y despues se "desenderaza" y se empalma
en la reconstruccion global (fuera de la ventana, se deja la reconstruccion
global tal cual: ahi DIP ya anda bien, el problema es solo la frontera).

Dos pasos, separados porque el DIP de la ventana corre en el cluster:

  prepare  -- arma la imagen y la mascara "enderezadas y recortadas"
              (analysis/windowed_frontier_dip.py prepare <run_dir> [--width 64])
  splice   -- una vez que el DIP de la ventana termino, la "desenderaza" y la
              empalma en la reconstruccion global; imprime PSNR/SSIM
              (analysis/windowed_frontier_dip.py splice <run_dir> <window_dir>)
"""
from __future__ import print_function

import argparse
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
    crop_pad, load_gray, load_original, row_crossings, smooth_fill,
)

PAD = 96  # relleno (replicando el borde) antes de correr filas, para no salir del cuadro


def per_row_shift(xs_smooth, center):
    """shift entero (para indexar) por fila: shift[y] = center - xs_smooth[y]."""
    return np.round(center - xs_smooth).astype(int)


def shift_rows_int(img, shift, pad_value):
    """Corre cada fila `shift[y]` columnas (entero), rellenando con pad_value."""
    H, W = img.shape
    padded = np.full((H, W + 2 * PAD), pad_value, dtype=img.dtype)
    padded[:, PAD:PAD + W] = img
    out = np.empty_like(padded)
    for y in range(H):
        out[y] = np.roll(padded[y], shift[y])
    return out


def prepare(run_dir, width):
    restored = crop_pad(load_gray(os.path.join(run_dir, "restored.png")))
    xs = smooth_fill(row_crossings(restored), 9)
    # filas de borde sin curva detectada: usar el valor valido mas cercano
    good = ~np.isnan(xs)
    idx = np.arange(len(xs))
    xs = np.interp(idx, idx[good], xs[good])

    H, W = restored.shape
    center = PAD + width // 2
    shift = per_row_shift(xs, center - PAD)

    orig_path = os.path.join(run_dir, "..", "..", "..", "..")  # no se usa: la imagen sale de IMAGE_PATH real
    # La imagen "real" (densa) que uso dip.restoration como target la tenemos
    # via final_comparison.png de esta misma corrida (columna 0), y la
    # mascara real via el .npy que uso esa corrida (buscado por convencion).
    image = load_original(run_dir)
    mask_path = _find_mask_path(run_dir)
    mask = np.load(mask_path)

    image_shift = shift_rows_int(image, shift, pad_value=float(np.median(image)))
    mask_shift = shift_rows_int(mask.astype(np.uint8), shift, pad_value=0).astype(bool)

    lo = PAD  # tras el shift, la ventana centrada queda siempre en [PAD, PAD+width)
    img_win = image_shift[:, lo:lo + width]
    mask_win = mask_shift[:, lo:lo + width]

    out_dir = os.path.join(run_dir, "window_prep")
    os.makedirs(out_dir, exist_ok=True)
    Image.fromarray((np.clip(img_win, 0, 1) * 255).astype(np.uint8), mode="L").save(
        os.path.join(out_dir, "image_window.png")
    )
    np.save(os.path.join(out_dir, "mask_window.npy"), mask_win)
    np.save(os.path.join(out_dir, "shift.npy"), shift)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"width": width, "center": center, "pad": PAD, "run_dir": run_dir}, f)
    print("Ventana: %d observados de %d (%.1f%%) -- antes %d de %d (%.1f%%)" % (
        mask_win.sum(), mask_win.size, 100 * mask_win.mean(),
        mask.sum(), mask.size, 100 * mask.mean(),
    ))
    print("Escrito", out_dir)


def _find_mask_path(run_dir):
    """Busca el .npy de mascara real que uso esta corrida, via metrics.csv no
    alcanza -- se infiere del nombre de carpeta (g<val>/mf<frac>/<family>)."""
    parts = os.path.normpath(run_dir).split(os.sep)
    family, mf_part, g_part = parts[-1], parts[-2], parts[-3]
    g = g_part[1:]
    mf = mf_part[2:]
    for base in ("data/restoration/masks_g", "data/restoration/masks_active"):
        cand = os.path.join(base, "g%s" % g, "mf%s" % mf, "mask_%s.npy" % family)
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError("no encontre la mascara real para " + run_dir)


def splice(run_dir, window_dir, out_suffix="windowed"):
    meta = json.load(open(os.path.join(run_dir, "window_prep", "meta.json")))
    width, center = meta["width"], meta["center"]
    shift = np.load(os.path.join(run_dir, "window_prep", "shift.npy"))

    restored_global = crop_pad(load_gray(os.path.join(run_dir, "restored.png")))
    original = load_original(run_dir)
    win_rec = crop_pad(load_gray(os.path.join(window_dir, "restored.png")))
    # win_rec puede tener alto distinto si dip.restoration recorto por crop_image(,64);
    # se asume 128 de alto (multiplo de 64), igual que la imagen madre.

    H, W = restored_global.shape
    lo = PAD
    spliced_padded = np.full((H, W + 2 * PAD), 0.0)
    spliced_padded[:, PAD:PAD + W] = restored_global
    for y in range(H):
        row = np.roll(spliced_padded[y], shift[y])
        row[lo:lo + width] = win_rec[y]
        spliced_padded[y] = np.roll(row, -shift[y])
    spliced = spliced_padded[:, PAD:PAD + W]
    spliced = np.clip(spliced, 0, 1)

    def ps(a, b):
        return dip_metrics.psnr(a[None], b[None]), dip_metrics.ssim(a[None], b[None], 1)

    p0, s0 = ps(original, restored_global)
    p1, s1 = ps(original, spliced)
    print("Global solo:      PSNR %.2f  SSIM %.4f" % (p0, s0))
    print("Con ventana DIP:  PSNR %.2f  SSIM %.4f" % (p1, s1))
    print("Delta: PSNR %+.2f  SSIM %+.4f" % (p1 - p0, s1 - s0))

    out_png = os.path.join(run_dir, "spliced_%s.png" % out_suffix)
    Image.fromarray((spliced * 255).astype(np.uint8), mode="L").save(out_png)

    fig, axs = plt.subplots(1, 3, figsize=(10.5, 4))
    axs[0].imshow(original, cmap="gray"); axs[0].set_title("Original")
    axs[1].imshow(restored_global, cmap="gray")
    axs[1].set_title("Global (PSNR %.1f, SSIM %.3f)" % (p0, s0))
    axs[2].imshow(spliced, cmap="gray")
    axs[2].set_title("Con ventana (PSNR %.1f, SSIM %.3f)" % (p1, s1))
    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(run_dir)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    cmp_png = os.path.join(run_dir, "window_comparison_%s.png" % out_suffix)
    fig.savefig(cmp_png, dpi=130)
    plt.close(fig)
    print("Escrito", out_png, "y", cmp_png)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("prepare")
    p1.add_argument("run_dir")
    p1.add_argument("--width", type=int, default=64)
    p2 = sub.add_parser("splice")
    p2.add_argument("run_dir")
    p2.add_argument("window_dir")
    args = ap.parse_args()
    if args.cmd == "prepare":
        prepare(args.run_dir, args.width)
    else:
        splice(args.run_dir, args.window_dir)
