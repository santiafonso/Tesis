#!/usr/bin/env python
# coding: utf-8
"""Ventana cuadrada (64x64, el minimo que acepta dip.restoration) centrada en
el "codo" de la frontera -- version simple, sin enderezar la imagen: un
recorte axis-aligned directo, a diferencia de analysis/windowed_frontier_dip.py
(que endereza y recorta una franja siguiendo TODA la curva).

  prepare  -- ubica el codo (maxima curvatura de la frontera suavizada) y
              recorta el cuadrado; guarda una preview con el marco
  splice   -- pega la reconstruccion del cuadrado en la reconstruccion global
              y reporta PSNR/SSIM antes/despues

Uso:
    ./venv/bin/python -m analysis.window_square prepare <run_dir> [--size 64]
    ./venv/bin/python -m analysis.window_square splice <run_dir> <window_dir>
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
import scipy.ndimage as ndi
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dip import metrics as dip_metrics  # noqa: E402
from analysis.correct_frontier_line import (  # noqa: E402
    crop_pad, load_gray, load_original, row_crossings, smooth_fill,
)
from analysis.windowed_frontier_dip import _find_mask_path  # noqa: E402


def find_elbow(xs_smooth, margin=8):
    good = ~np.isnan(xs_smooth)
    idx = np.where(good)[0]
    curve = ndi.gaussian_filter1d(xs_smooth[good], sigma=3)
    d2 = np.gradient(np.gradient(curve))
    inner = slice(margin, len(idx) - margin)
    k_rel = np.argmax(np.abs(d2[inner])) + margin
    k = idx[k_rel]
    return k, xs_smooth[k]  # (fila, x)


def prepare(run_dir, size):
    restored = crop_pad(load_gray(os.path.join(run_dir, "restored.png")))
    original = load_original(run_dir)
    xs = row_crossings(restored)
    xs_s = smooth_fill(xs, 9)
    cy, cx = find_elbow(xs_s)

    H, W = original.shape
    half = size // 2
    r0 = int(np.clip(round(cy - half), 0, H - size))
    c0 = int(np.clip(round(cx - half), 0, W - size))

    mask = np.load(_find_mask_path(run_dir))
    img_win = original[r0:r0 + size, c0:c0 + size]
    mask_win = mask[r0:r0 + size, c0:c0 + size]

    out_dir = os.path.join(run_dir, "window_sq%d" % size)
    os.makedirs(out_dir, exist_ok=True)
    Image.fromarray((np.clip(img_win, 0, 1) * 255).astype(np.uint8), mode="L").save(
        os.path.join(out_dir, "image_window.png")
    )
    np.save(os.path.join(out_dir, "mask_window.npy"), mask_win)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"size": size, "r0": r0, "c0": c0, "cy": float(cy), "cx": float(cx)}, f)

    print("Codo en fila=%d x=%.1f -> ventana [%d:%d, %d:%d]  (%d obs de %d, %.1f%%)" % (
        cy, cx, r0, r0 + size, c0, c0 + size, mask_win.sum(), mask_win.size,
        100 * mask_win.mean(),
    ))

    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    ax.imshow(original, cmap="gray")
    ax.add_patch(plt.Rectangle((c0, r0), size, size, fill=False, edgecolor="red", linewidth=2))
    ax.plot([cx], [cy], "x", color="yellow", ms=10, mew=2)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Ventana %dx%d en [%d:%d, %d:%d]" % (size, size, r0, r0 + size, c0, c0 + size))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "frame_preview.png"), dpi=130)
    plt.close(fig)
    print("Escrito", out_dir)


def splice(run_dir, window_dir):
    base = os.path.basename(window_dir)
    prep_dir = os.path.join(run_dir, base[:base.index("_dip")])  # window_sq64_dip[_reg0.08] -> window_sq64
    meta = json.load(open(os.path.join(prep_dir, "meta.json")))
    size, r0, c0 = meta["size"], meta["r0"], meta["c0"]

    restored_global = crop_pad(load_gray(os.path.join(run_dir, "restored.png")))
    original = load_original(run_dir)
    win_rec = crop_pad(load_gray(os.path.join(window_dir, "restored.png")))

    spliced = restored_global.copy()
    spliced[r0:r0 + size, c0:c0 + size] = win_rec
    spliced = np.clip(spliced, 0, 1)

    def ps(a, b):
        return dip_metrics.psnr(a[None], b[None]), dip_metrics.ssim(a[None], b[None], 1)

    p0, s0 = ps(original, restored_global)
    p1, s1 = ps(original, spliced)
    print("Global solo:      PSNR %.2f  SSIM %.4f" % (p0, s0))
    print("Con ventana sq:   PSNR %.2f  SSIM %.4f" % (p1, s1))
    print("Delta: PSNR %+.2f  SSIM %+.4f" % (p1 - p0, s1 - s0))

    tag = base[base.index("_dip") + 5:] or "sq%d" % size  # sufijo distintivo (p.ej. reg0.08)
    tag = "sq%d_%s" % (size, tag) if tag != "sq%d" % size else tag

    Image.fromarray((spliced * 255).astype(np.uint8), mode="L").save(
        os.path.join(run_dir, "spliced_%s.png" % tag)
    )

    fig, axs = plt.subplots(1, 3, figsize=(10.5, 4))
    axs[0].imshow(original, cmap="gray"); axs[0].set_title("Original")
    axs[1].imshow(restored_global, cmap="gray")
    axs[1].set_title("Global (PSNR %.1f, SSIM %.3f)" % (p0, s0))
    for ax in axs[:2]:
        ax.add_patch(plt.Rectangle((c0, r0), size, size, fill=False, edgecolor="red", linewidth=1.5))
    axs[2].imshow(spliced, cmap="gray")
    axs[2].set_title("Con ventana sq (PSNR %.1f, SSIM %.3f)" % (p1, s1))
    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(run_dir)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(run_dir, "window_comparison_%s.png" % tag), dpi=130)
    plt.close(fig)
    print("Escrito spliced_%s.png y window_comparison_%s.png" % (tag, tag))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("prepare")
    p1.add_argument("run_dir")
    p1.add_argument("--size", type=int, default=64)
    p2 = sub.add_parser("splice")
    p2.add_argument("run_dir")
    p2.add_argument("window_dir")
    args = ap.parse_args()
    if args.cmd == "prepare":
        prepare(args.run_dir, args.size)
    else:
        splice(args.run_dir, args.window_dir)
