#!/usr/bin/env python
# coding: utf-8
"""Prueba de robustez 'fuera de familia': el punto que hizo notar el usuario
es que en el caso real NO vamos a saber si la imagen viene exactamente del
modelo continuo -- puede haber desviaciones (ruido KMC real, aproximaciones
del continuo) que ni g ni vcut expliquen. Esta prueba simula eso a proposito:

  1. Toma un mapa real (g conocido) y le SUMA una perturbacion suave 2D
     (una onda de baja frecuencia) que el modelo con solo (g, vcut) libres
     NO puede reproducir -- una desviacion "fuera de familia" a proposito.
  2. Calibra g igual que antes, pero ahora contra el mapa YA perturbado.
  3. Mide que tan mal queda si uno confia CIEGAMENTE en el modelo calibrado
     (sin corregir), vs. un HIBRIDO: usar el modelo calibrado como base y
     corregir con el RESIDUO (observado - modelo) interpolado suavemente
     entre los mismos puntos dispersos (nucleo gaussiano, misma idea que
     `curvefit.fit_boundary_curve` pero aplicada al residuo en vez de a la
     posicion de la frontera).

Corre a resolucion reducida (32x32) para que sea rapido -- es una prueba de
CONCEPTO de robustez, no el resultado final.

Uso:
    ./venv/bin/python -m physics_fit.robustness_test -4.0 60
"""
from __future__ import print_function

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dip import metrics as dip_metrics  # noqa: E402
import physics_fit.calibrate as pf  # noqa: E402

RES = 32


def downsample(img, factor):
    H, W = img.shape
    h2, w2 = H // factor, W // factor
    return img[:h2 * factor, :w2 * factor].reshape(h2, factor, w2, factor).mean(axis=(1, 3))


def kernel_smooth_2d(rows, cols, vals, H, W, bandwidth=6.0):
    """Interpolacion por nucleo gaussiano 2D -- version 2D de
    `curvefit.fit_boundary_curve`, para suavizar el RESIDUO disperso a una
    grilla completa."""
    yy, xx = np.mgrid[0:H, 0:W]
    out = np.zeros((H, W), dtype=np.float64)
    wsum = np.zeros((H, W), dtype=np.float64)
    for r, c, v in zip(rows, cols, vals):
        d2 = (yy - r) ** 2 + (xx - c) ** 2
        k = np.exp(-0.5 * d2 / bandwidth ** 2)
        out += k * v
        wsum += k
    return out / np.where(wsum > 1e-9, wsum, 1.0)


def main():
    g_real = sys.argv[1] if len(sys.argv) > 1 else "-4.0"
    n_pts = int(sys.argv[2]) if len(sys.argv) > 2 else 60

    gt128 = pf.load_ground_truth(g_real)
    gt32 = downsample(gt128, 128 // RES)

    # --- perturbacion "fuera de familia": onda suave 2D, amplitud 0.12 ---
    yy, xx = np.mgrid[0:RES, 0:RES]
    perturb = 0.12 * np.sin(2 * np.pi * yy / RES * 1.5) * np.cos(2 * np.pi * xx / RES * 2.0)
    gt_perturbed = np.clip(gt32 + perturb, 0, 1).astype(np.float32)

    rows, cols = pf.sample_points(RES, RES, n_pts)
    observed = gt_perturbed[rows, cols]
    print("g real=%s (+ perturbacion fuera de familia), %d puntos" % (g_real, len(rows)), flush=True)

    t0 = time.time()
    g_hat, n_eval = pf.fit_g(rows, cols, observed, H=RES, W=RES)
    print("g_hat=%.4f (%d evals, %.1f s)" % (g_hat, n_eval, time.time() - t0), flush=True)

    t0 = time.time()
    model_only = pf.regenerate_full_map(g_hat, H=RES, W=RES)
    print("Modelo calibrado (sin correccion) regenerado en %.1f s" % (time.time() - t0), flush=True)

    # residuo en los puntos observados, interpolado a toda la grilla, sumado
    # al modelo calibrado -- el hibrido.
    model_at_pts = model_only[rows, cols]
    residual_obs = observed - model_at_pts
    residual_full = kernel_smooth_2d(rows, cols, residual_obs, RES, RES)
    hybrid = np.clip(model_only + residual_full, 0, 1)

    def metr(rec, name):
        psnr = dip_metrics.psnr(gt_perturbed[None], rec[None])
        ssim = dip_metrics.ssim(gt_perturbed[None], rec[None], 1)
        mae = dip_metrics.mae(gt_perturbed[None], rec[None])
        print("%-22s PSNR %.2f dB  SSIM %.4f  MAE %.4f" % (name, psnr, ssim, mae), flush=True)
        return psnr, ssim, mae

    metr(model_only, "solo modelo (g_hat):")
    metr(hybrid, "hibrido (+ residuo):")

    out_dir = os.path.join(pf.OUT_ROOT, "robustness_g%s_n%d" % (g_real, n_pts))
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "arrays.npz"), gt=gt_perturbed, model_only=model_only,
             hybrid=hybrid, rows=rows, cols=cols)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 4, figsize=(14, 4))
    axs[0].imshow(gt_perturbed, cmap="gray", vmin=0, vmax=1)
    axs[0].scatter(cols, rows, s=10, c="red")
    axs[0].set_title("Real (fuera de familia) + puntos")
    axs[1].imshow(model_only, cmap="gray", vmin=0, vmax=1)
    axs[1].set_title("Solo modelo (g_hat=%.2f)" % g_hat)
    axs[2].imshow(hybrid, cmap="gray", vmin=0, vmax=1)
    axs[2].set_title("Hibrido (+ residuo)")
    axs[3].imshow(np.abs(gt_perturbed - hybrid), cmap="inferno", vmin=0, vmax=0.3)
    axs[3].set_title("|error| hibrido")
    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "comparison.png"), dpi=130)
    plt.close(fig)
    print("Escrito", os.path.join(out_dir, "comparison.png"), flush=True)


if __name__ == "__main__":
    main()
