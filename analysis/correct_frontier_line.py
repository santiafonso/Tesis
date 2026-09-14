#!/usr/bin/env python
# coding: utf-8
"""Correccion de la reconstruccion DIP para g muy negativo: en vez de solo
mostrar la frontera real encima (overlay, lo que ya se hizo), esto CORRIGE
la reconstruccion usando unicamente su propia salida -- sin ver la imagen
real, por eso si es desplegable con puntos KMC reales:

  1. Por cada fila, se detecta donde la reconstruccion cruza el nivel 0.5
     (sub-pixel, por interpolacion) -- una curva ruidosa, con el patron de
     escalones que ya se vio en los montajes de g muy negativo.
  2. Se suaviza esa curva con una media movil ("trazar una media").
  3. Se corre cada fila lateralmente (interpolacion) para que su cruce quede
     sobre la posicion suavizada -- "endereza" el escalon.

Uso:
    ./venv/bin/python -m analysis.correct_frontier_line \
        results/dip_gsweep_frontier/g-4.0/mf0.980/uniform

Escribe <dir>/corrected.png y <dir>/correction_comparison.png, e imprime
PSNR/SSIM antes/despues.
"""
from __future__ import print_function

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dip import metrics as dip_metrics  # noqa: E402

SMOOTH_WINDOW = int(os.environ.get("SMOOTH_WINDOW", "9"))


def crop_pad(img):
    """Saca el ReflectionPad2d(1): imagen NxN -> (N-2)x(N-2)."""
    return img[1:-1, 1:-1]


def load_gray(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def load_original(run_dir):
    fc = Image.open(os.path.join(run_dir, "final_comparison.png"))
    w = fc.width // 4
    orig = np.asarray(fc.crop((0, 0, w, fc.height)).convert("L"), dtype=np.float32) / 255.0
    return crop_pad(orig)


def row_crossings(img, level=0.5):
    """Para cada fila, x (sub-pixel) donde img cruza `level` yendo de alto a
    bajo (izquierda a derecha). NaN si esa fila no cruza."""
    H, W = img.shape
    xs = np.full(H, np.nan)
    for y in range(H):
        row = img[y]
        below = row < level
        if not below.any() or below.all():
            continue
        k = np.argmax(below)  # primer indice por debajo del nivel
        if k == 0:
            xs[y] = 0.0
            continue
        v0, v1 = row[k - 1], row[k]
        frac = (level - v0) / (v1 - v0) if v1 != v0 else 0.5
        xs[y] = (k - 1) + frac
    return xs


def smooth_fill(xs, window):
    """Media movil. Los NaN INTERIORES (rodeados de cruces validos, por el
    escalonado) se interpolan antes de suavizar; los de los bordes (filas
    donde el frente ya salio del cuadro y no hay cruce real) se dejan como
    NaN -- interpolarlos inventaria un cruce que no existe y desvia la media
    movil de golpe."""
    idx = np.arange(len(xs))
    good = ~np.isnan(xs)
    if good.sum() < 2:
        return xs
    first, last = idx[good][0], idx[good][-1]
    filled = xs.copy()
    interior = slice(first, last + 1)
    filled[interior] = np.interp(idx[interior], idx[good], xs[good])
    smoothed = uniform_filter1d(filled[interior], size=window, mode="nearest")
    out = np.full_like(xs, np.nan)
    out[interior] = smoothed
    return out


def shift_rows(img, shift):
    """corrected[y, x] = img[y, x - shift[y]] (interpolado)."""
    H, W = img.shape
    out = np.empty_like(img)
    xs = np.arange(W)
    for y in range(H):
        out[y] = np.interp(xs - shift[y], xs, img[y])
    return out


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else \
        "results/dip_gsweep_frontier/g-4.0/mf0.980/uniform"
    restored = crop_pad(load_gray(os.path.join(run_dir, "restored.png")))
    original = load_original(run_dir)

    xs_rec = row_crossings(restored)
    xs_true = row_crossings(original)
    xs_smooth = smooth_fill(xs_rec, SMOOTH_WINDOW)
    shift = xs_smooth - xs_rec
    shift[np.isnan(xs_rec) | np.isnan(xs_smooth)] = 0.0

    corrected = shift_rows(restored, shift)
    corrected = np.clip(corrected, 0, 1)

    psnr_before = dip_metrics.psnr(original[None], restored[None])
    ssim_before = dip_metrics.ssim(original[None], restored[None], 1)
    psnr_after = dip_metrics.psnr(original[None], corrected[None])
    ssim_after = dip_metrics.ssim(original[None], corrected[None], 1)

    print("PSNR antes:  %.2f dB   SSIM antes:  %.4f" % (psnr_before, ssim_before))
    print("PSNR despues: %.2f dB   SSIM despues: %.4f" % (psnr_after, ssim_after))
    print("Delta PSNR: %+.2f dB    Delta SSIM: %+.4f" % (psnr_after - psnr_before, ssim_after - ssim_before))

    Image.fromarray((corrected * 255).astype(np.uint8), mode="L").save(
        os.path.join(run_dir, "corrected.png")
    )

    fig, axs = plt.subplots(1, 4, figsize=(14, 4))
    axs[0].imshow(original, cmap="gray")
    axs[0].set_title("Original")
    axs[1].imshow(restored, cmap="gray")
    y = np.arange(len(xs_rec))
    axs[1].plot(xs_rec, y, ".", color="red", ms=2, alpha=0.6)
    axs[1].set_title("DIP (PSNR %.1f, SSIM %.3f)\nrojo=cruce detectado" % (psnr_before, ssim_before))
    axs[2].imshow(restored, cmap="gray")
    axs[2].plot(xs_rec, y, ".", color="red", ms=2, alpha=0.4, label="detectado")
    axs[2].plot(xs_smooth, y, "-", color="lime", linewidth=1.5, label="media movil")
    axs[2].legend(fontsize=8, loc="lower right")
    axs[2].set_title("Suavizado")
    axs[3].imshow(corrected, cmap="gray")
    axs[3].set_title("Corregida (PSNR %.1f, SSIM %.3f)" % (psnr_after, ssim_after))
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(run_dir)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_png = os.path.join(run_dir, "correction_comparison.png")
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print("Escrito", out_png)


if __name__ == "__main__":
    main()
