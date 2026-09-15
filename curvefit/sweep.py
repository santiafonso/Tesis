#!/usr/bin/env python
# coding: utf-8
"""Compara `curvefit.reconstruct` (sin red neuronal) contra `dip.restoration`
en el mismo barrido de 9 g x 5 mf x 5 familias del punto 3 de la reunion del
9/9 (ver `analysis/gsweep_5family_summary.py`) -- misma mascara, mismo ground
truth (`soc.npy` real), asi que las metricas son comparables cara a cara.

El PSNR/SSIM/MAE de DIP se RECALCULA desde `restored.png` contra `soc.npy`
(no se reusa `metrics.csv`, que esta calculado contra el PNG de 8 bits
`sim_128.png` -- una fuente de verdad ligeramente distinta).

Escribe:
    results/curvefit_gsweep/summary.csv
    results/curvefit_gsweep/psnr_vs_pct_by_g.png   (grilla 3x3, solido=DIP,
                                                     punteado=curve-fit)

Uso:
    ./venv/bin/python -m curvefit.sweep
"""
from __future__ import print_function

import csv
import os
import sys
import traceback

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dip import metrics as dip_metrics  # noqa: E402
from analysis.correct_frontier_line import crop_pad, load_gray  # noqa: E402
from curvefit import reconstruct as cf  # noqa: E402

GS = [-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0]
MFS = ["0.900", "0.950", "0.980", "0.990", "0.995"]
FAMILIES = ["uniform", "grid", "spread", "frontier", "frontier_mix"]
DIP_DIR = "results/dip_gsweep_frontier"
OUT_DIR = "results/curvefit_gsweep"

COLORS = {
    "uniform": "#7f7f7f", "grid": "#1f77b4", "spread": "#2ca02c",
    "frontier": "#d62728", "frontier_mix": "#ff7f0e",
}
MARKERS = {
    "uniform": "o", "grid": "s", "spread": "v", "frontier": "D", "frontier_mix": "^",
}


def dip_metrics_from_restored(g, mf, family):
    path = os.path.join(DIP_DIR, "g%s" % g, "mf%s" % mf, family, "restored.png")
    if not os.path.exists(path):
        return None
    gt = cf.load_ground_truth(g)
    rec = crop_pad(load_gray(path))
    if rec.shape != gt.shape:
        return None
    return {
        "psnr": float(dip_metrics.psnr(gt[None], rec[None])),
        "ssim": float(dip_metrics.ssim(gt[None], rec[None], 1)),
        "mae": float(dip_metrics.mae(gt[None], rec[None])),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    records = []
    for g in GS:
        for mf in MFS:
            pct_obs = 100.0 * (1.0 - float(mf))
            for fam in FAMILIES:
                rec = {"g": g, "mf": mf, "pct_obs": pct_obs, "family": fam}
                try:
                    cfr = cf.reconstruct(str(g), mf, fam)
                    rec.update({
                        "n_obs": cfr["n_obs"],
                        "psnr_curvefit": cfr["psnr"],
                        "ssim_curvefit": cfr["ssim"],
                        "mae_curvefit": cfr["mae"],
                    })
                except Exception as exc:
                    print("!! curvefit fallo g=%s mf=%s %s: %s" % (g, mf, fam, exc))
                    traceback.print_exc()
                    rec.update({"n_obs": None, "psnr_curvefit": float("nan"),
                                "ssim_curvefit": float("nan"), "mae_curvefit": float("nan")})

                dm = dip_metrics_from_restored(g, mf, fam)
                if dm is None:
                    print("!! falta restored.png de DIP para g=%s mf=%s %s" % (g, mf, fam))
                    rec.update({"psnr_dip": float("nan"), "ssim_dip": float("nan"), "mae_dip": float("nan")})
                else:
                    rec.update({"psnr_dip": dm["psnr"], "ssim_dip": dm["ssim"], "mae_dip": dm["mae"]})
                records.append(rec)

    csv_path = os.path.join(OUT_DIR, "summary.csv")
    fieldnames = ["g", "mf", "pct_obs", "family", "n_obs",
                  "psnr_curvefit", "ssim_curvefit", "mae_curvefit",
                  "psnr_dip", "ssim_dip", "mae_dip"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(records)
    print("Escrito", csv_path, "(%d filas)" % len(records))

    fig, axs = plt.subplots(3, 3, figsize=(16, 13), sharex=True, sharey=False)
    for ax, g in zip(axs.flat, GS):
        for fam in FAMILIES:
            sub = sorted([r for r in records if r["g"] == g and r["family"] == fam],
                         key=lambda r: r["pct_obs"])
            xs = [r["pct_obs"] for r in sub]
            ax.plot(xs, [r["psnr_dip"] for r in sub], marker=MARKERS[fam], color=COLORS[fam],
                     linewidth=1.8, markersize=5, linestyle="-", label="%s (DIP)" % fam)
            ax.plot(xs, [r["psnr_curvefit"] for r in sub], marker=MARKERS[fam], color=COLORS[fam],
                     linewidth=1.3, markersize=4, linestyle="--", alpha=0.75,
                     label="%s (curve-fit)" % fam)
        ax.set_title("g = %s" % g, fontsize=11)
        ax.set_xscale("log")
        ax.grid(alpha=0.3)
    for ax in axs[-1, :]:
        ax.set_xlabel("% observado")
    for ax in axs[:, 0]:
        ax.set_ylabel("PSNR final (dB)")
    fig.suptitle(
        "curve-fit (sin red neuronal, punteado) vs. DIP (solido) -- PSNR vs. %% observado, g negativo\n"
        "(mismo ground truth soc.npy y mismas mascaras para ambos)",
        fontsize=13, y=0.995,
    )
    handles, labels = axs.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, 0.955))
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    png_path = os.path.join(OUT_DIR, "psnr_vs_pct_by_g.png")
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    print("Escrito", png_path)


if __name__ == "__main__":
    main()
