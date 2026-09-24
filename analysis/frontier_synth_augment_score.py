#!/usr/bin/env python
# coding: utf-8
"""Puntaje honesto (baseline vs. augmented) del experimento de puntos
sinteticos de meseta (`analysis/frontier_synth_augment.py` +
`slurm/dip_synth_augment_test.slurm`): compara `restored.npy` contra el
soc.npy REAL, no contra la imagen compuesta (que tiene valores inventados
en los puntos sinteticos -- puntuar contra ella haria trampa).

Uso:
    ./venv/bin/python -m analysis.frontier_synth_augment_score -4.0 0.995 frontier_mix
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dip import metrics as dip_metrics  # noqa: E402
from utils.common_utils import crop_image  # noqa: E402

PHASE_DIAGRAM_DIR = "results/phase_diagram_g"
OUT_ROOT = "results/experimentos_menos_puntos/synth_augment"


def load_ground_truth_cropped(g, ref_shape):
    """soc.npy volteado (misma convencion que el resto) y recortado/redim.
    para calzar con el `restored.npy` de dip.restoration (que recorta a
    multiplo de 64 antes del reflection-pad)."""
    soc = np.flipud(np.load(os.path.join(PHASE_DIAGRAM_DIR, "g%s" % g, "soc.npy"))).astype(np.float32)
    if soc.shape != ref_shape:
        soc = np.asarray(
            Image.fromarray((np.clip(soc, 0, 1) * 255).astype(np.uint8)).resize(
                (ref_shape[1], ref_shape[0]), Image.BILINEAR),
            dtype=np.float32,
        ) / 255.0
    return soc


def crop_pad(img):
    return img[1:-1, 1:-1]


def main():
    g, mf, family = sys.argv[1], sys.argv[2], sys.argv[3]
    base_dir = os.path.join(OUT_ROOT, "g%s" % g, "mf%s" % mf, family)

    results = {}
    for name in ["baseline", "augmented"]:
        restored = np.load(os.path.join(base_dir, name, "restored.npy"))
        if restored.ndim == 3:
            restored = restored[0]
        restored = crop_pad(restored)
        gt = load_ground_truth_cropped(g, restored.shape)
        psnr = dip_metrics.psnr(gt[None], restored[None])
        ssim = dip_metrics.ssim(gt[None], restored[None], 1)
        mae = dip_metrics.mae(gt[None], restored[None])
        results[name] = {"restored": restored, "gt": gt, "psnr": psnr, "ssim": ssim, "mae": mae}
        print("%-10s PSNR=%.2f dB  SSIM=%.4f  MAE=%.4f" % (name, psnr, ssim, mae))

    delta_psnr = results["augmented"]["psnr"] - results["baseline"]["psnr"]
    delta_ssim = results["augmented"]["ssim"] - results["baseline"]["ssim"]
    print("Delta (augmented - baseline): PSNR %+.2f dB   SSIM %+.4f" % (delta_psnr, delta_ssim))

    gt = results["baseline"]["gt"]
    fig, axs = plt.subplots(1, 3, figsize=(11, 4))
    axs[0].imshow(gt, cmap="gray", vmin=0, vmax=1)
    axs[0].set_title("Original")
    axs[1].imshow(results["baseline"]["restored"], cmap="gray", vmin=0, vmax=1)
    axs[1].set_title("Solo reales (PSNR %.1f, SSIM %.3f)" % (
        results["baseline"]["psnr"], results["baseline"]["ssim"]))
    axs[2].imshow(results["augmented"]["restored"], cmap="gray", vmin=0, vmax=1)
    axs[2].set_title("+ sinteticos (PSNR %.1f, SSIM %.3f)" % (
        results["augmented"]["psnr"], results["augmented"]["ssim"]))
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("g=%s mf=%s %s -- puntaje contra soc.npy real (no la imagen compuesta)" % (g, mf, family))
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_png = os.path.join(base_dir, "comparison.png")
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print("Escrito", out_png)


if __name__ == "__main__":
    main()
