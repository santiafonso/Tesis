#!/usr/bin/env python
# coding: utf-8
"""Corrida piloto completa (fidelidad real, no la version rapida de smoke
test): ajusta g con ~60 puntos a fidelidad default, y regenera el mapa a
64x64 (128x128 tardaria ~1.5-2h localmente segun lo medido -- ver
CLAUDE.md, necesitaria cluster) para poder compararlo rapido contra el
ground truth verdadero (downsampleado a 64x64 por promedio de bloques, sin
volver a correr el modelo)."""
from __future__ import print_function

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dip import metrics as dip_metrics  # noqa: E402
import physics_fit.calibrate as pf  # noqa: E402


def downsample(img, factor):
    H, W = img.shape
    h2, w2 = H // factor, W // factor
    return img[:h2 * factor, :w2 * factor].reshape(h2, factor, w2, factor).mean(axis=(1, 3))


def main():
    g_real = sys.argv[1] if len(sys.argv) > 1 else "-4.0"
    n_pts = int(sys.argv[2]) if len(sys.argv) > 2 else 60

    gt128 = pf.load_ground_truth(g_real)
    rows, cols = pf.sample_points(pf.H, pf.W, n_pts)
    observed = gt128[rows, cols]
    print("g real=%s, %d puntos observados" % (g_real, len(rows)), flush=True)

    t0 = time.time()
    g_hat, n_eval = pf.fit_g(rows, cols, observed)
    fit_time = time.time() - t0
    print("\ng_hat=%.4f (real=%s), %d evaluaciones, %.1f s" % (g_hat, g_real, n_eval, fit_time), flush=True)

    t0 = time.time()
    rec64 = pf.regenerate_full_map(g_hat, H=64, W=64)
    gen_time = time.time() - t0
    print("Mapa regenerado a 64x64 en %.1f s" % gen_time, flush=True)

    gt64 = downsample(gt128, 2)
    psnr = dip_metrics.psnr(gt64[None], rec64[None])
    ssim = dip_metrics.ssim(gt64[None], rec64[None], 1)
    mae = dip_metrics.mae(gt64[None], rec64[None])
    print("A 64x64 -- PSNR: %.2f dB   SSIM: %.4f   MAE: %.4f" % (psnr, ssim, mae), flush=True)

    out_dir = os.path.join(pf.OUT_ROOT, "g%s" % g_real, "n%d_pilot" % n_pts)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "rec64.npy"), rec64)
    np.save(os.path.join(out_dir, "gt64.npy"), gt64)
    np.save(os.path.join(out_dir, "rows.npy"), rows)
    np.save(os.path.join(out_dir, "cols.npy"), cols)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write("g_real=%s g_hat=%.4f n_pts=%d n_eval=%d fit_time=%.1f gen_time=%.1f "
                "psnr=%.2f ssim=%.4f mae=%.4f\n" % (
                    g_real, g_hat, n_pts, n_eval, fit_time, gen_time, psnr, ssim, mae))
    print("Escrito en", out_dir, flush=True)


if __name__ == "__main__":
    main()
