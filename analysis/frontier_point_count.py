#!/usr/bin/env python
# coding: utf-8
"""Punto 2 de la reunion del 9/9: para las corridas `uniform` (Bernoulli) del
barrido dip_gsweep_frontier, contar cuantos de sus puntos observados caen
sobre la frontera de fase y cruzarlo con la fidelidad final de esa
reconstruccion -- para ver si lo que predice una reconstruccion confiable es
el numero de puntos EN LA FRONTERA, no el total de puntos observados (mf).

Reusa/regenera data/restoration/masks_g/ (no esta versionada localmente: se
genero en el cluster) llamando a dip.frontier_mask con los mismos parametros
que slurm/frontier_mask_g.slurm, asi las mascaras son bit-a-bit las que se
usaron para producir resultss/dip_gsweep_frontier/ (mismo SEED=42).

Uso:
    ./venv/bin/python -m analysis.frontier_point_count

Escribe:
    results/frontier_point_count/summary.csv
    results/frontier_point_count/psnr_vs_frontier_points.png
"""
from __future__ import print_function

import csv
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from PIL import Image

GS = [-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0]
MFS = ["0.900", "0.950", "0.980", "0.990", "0.995"]
FAMILIES = ["uniform", "frontier_mix"]  # las que realmente corrieron en dip_gsweep_frontier
RES = 128
OBS_FRACS = "0.10 0.05 0.02 0.01 0.005"  # espeja frontier_mask_g.slurm -> mf 0.90..0.995
SEED = "42"
EDGE_THRESH = 0.30  # mismo umbral que el print "sobre frontera" de dip.frontier_mask
SMOOTH_SIGMA = 1.0

PHASE_G_DIR = "results/phase_diagram_g"
MASKS_G_DIR = "data/restoration/masks_g"
RESULTS_DIR = "results/dip_gsweep_frontier"
OUT_DIR = "results/frontier_point_count"

os.makedirs(OUT_DIR, exist_ok=True)


def ensure_masks(g):
    out = os.path.join(MASKS_G_DIR, "g%s" % g)
    rala = os.path.join(out, "mf0.995", "mask_uniform.npy")
    if os.path.exists(rala):
        return out
    image_path = os.path.join(PHASE_G_DIR, "g%s" % g, "sim_%d.png" % RES)
    soc_npy = os.path.join(PHASE_G_DIR, "g%s" % g, "soc.npy")
    if not os.path.exists(image_path):
        print("!! falta", image_path, "-- salteo g=%s" % g)
        return None
    if not os.path.exists(soc_npy):
        print("(sin soc.npy para g=%s, dip.frontier_mask usa el PNG como proxy)" % g)
    print("Regenerando mascaras para g=%s (no estaban en local) ..." % g)
    env = dict(os.environ)
    env.update({
        "IMAGE_PATH": image_path,
        "SOC_NPY": soc_npy,
        "OUT_DIR": out,
        "OBS_FRACS": OBS_FRACS,
        "SEED": SEED,
    })
    subprocess.run(
        [sys.executable, "-m", "dip.frontier_mask"], env=env, check=True,
        stdout=subprocess.DEVNULL,
    )
    return out


def edge_field_for(g):
    soc_npy = os.path.join(PHASE_G_DIR, "g%s" % g, "soc.npy")
    if os.path.exists(soc_npy):
        soc = np.load(soc_npy).astype(np.float32)
        soc = np.flipud(soc)  # misma convencion que dip.frontier_mask
    else:
        # mismo fallback que dip.frontier_mask cuando falta SOC_NPY: usar el
        # PNG de entrada tal cual (ya en la orientacion/crop que ve DIP).
        image_path = os.path.join(PHASE_G_DIR, "g%s" % g, "sim_%d.png" % RES)
        img_pil = Image.open(image_path).convert("L")
        soc = np.asarray(img_pil, dtype=np.float32) / 255.0
    gy, gx = np.gradient(ndi.gaussian_filter(soc, SMOOTH_SIGMA))
    gmag = np.hypot(gx, gy)
    gmag = ndi.gaussian_filter(gmag, SMOOTH_SIGMA)
    return gmag / max(gmag.max(), 1e-12)


def read_final_metrics(metrics_csv):
    if not os.path.exists(metrics_csv):
        return None
    rows = {}
    with open(metrics_csv) as f:
        for row in csv.reader(f):
            if not row or row[0] in ("iter", ""):
                continue
            rows[row[0]] = row
    final = rows.get("final")
    mae = rows.get("mae")
    fallbacks = rows.get("fallbacks")
    if final is None:
        return None
    return {
        "psnr_final": float(final[1]) if final[1] else np.nan,
        "ssim_final": float(final[3]) if final[3] else np.nan,
        "mae_final": float(mae[1]) if mae and mae[1] else np.nan,
        "fallbacks": int(float(fallbacks[1])) if fallbacks and fallbacks[1] else np.nan,
    }


def main():
    records = []
    for g in GS:
        masks_dir = ensure_masks(g)
        if masks_dir is None:
            continue
        edge = edge_field_for(g)
        for mf in MFS:
            for fam in FAMILIES:
                mask_path = os.path.join(masks_dir, "mf%s" % mf, "mask_%s.npy" % fam)
                metrics_path = os.path.join(RESULTS_DIR, "g%s" % g, "mf%s" % mf, fam, "metrics.csv")
                if not os.path.exists(mask_path):
                    print("!! falta mascara", mask_path)
                    continue
                m = np.load(mask_path)
                n = int(m.sum())
                n_on_edge = int((m & (edge > EDGE_THRESH)).sum())
                frac_on_edge = n_on_edge / max(n, 1)
                metrics = read_final_metrics(metrics_path)
                if metrics is None:
                    print("!! sin metrics.csv para", metrics_path)
                    continue
                rec = {"g": g, "mf": mf, "family": fam, "n_obs": n,
                       "n_on_edge": n_on_edge, "frac_on_edge": frac_on_edge}
                rec.update(metrics)
                records.append(rec)

    if not records:
        print("Sin registros -- revisa que results/phase_diagram_g y "
              "results/dip_gsweep_frontier existan.")
        return

    csv_path = os.path.join(OUT_DIR, "summary.csv")
    fieldnames = ["g", "mf", "family", "n_obs", "n_on_edge", "frac_on_edge",
                  "psnr_final", "ssim_final", "mae_final", "fallbacks"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(records)
    print("Escrito", csv_path, "(%d filas)" % len(records))

    # --- plot: fidelidad vs. puntos-en-frontera, separado por familia -----
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    gs_vals = sorted(set(r["g"] for r in records))
    cmap = plt.cm.viridis
    norm = plt.Normalize(min(gs_vals), max(gs_vals))
    markers = {"uniform": "o", "frontier_mix": "^"}

    for ax_row, ycol, ylabel in [(0, "psnr_final", "PSNR final (dB)"),
                                  (1, "ssim_final", "SSIM final")]:
        for ax_col, xcol, xlabel in [(0, "n_on_edge", "# puntos sobre la frontera"),
                                      (1, "frac_on_edge", "fraccion de puntos sobre la frontera")]:
            ax = axs[ax_row, ax_col]
            for fam in FAMILIES:
                sub = [r for r in records if r["family"] == fam]
                xs = [r[xcol] for r in sub]
                ys = [r[ycol] for r in sub]
                cs = [cmap(norm(r["g"])) for r in sub]
                ax.scatter(xs, ys, c=cs, marker=markers[fam], s=60,
                           edgecolors="k", linewidths=0.4,
                           label=fam if ax_row == 0 and ax_col == 0 else None)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axs, label="g", shrink=0.8)
    axs[0, 0].legend(loc="lower right", fontsize=9)
    fig.suptitle(
        "Punto 2: fidelidad final vs. cuantos puntos observados caen en la frontera\n"
        "(circulo=uniform, triangulo=frontier_mix; color=g)"
    )
    png_path = os.path.join(OUT_DIR, "psnr_vs_frontier_points.png")
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    print("Escrito", png_path)


if __name__ == "__main__":
    main()
