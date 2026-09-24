#!/usr/bin/env python
# coding: utf-8
"""Comparacion por g: metodo nuevo de autoexp (64 puntos, sin DIP) vs las mejores
reconstrucciones DIP previas de la tesis (frontier_mix + REG_NOISE_STD=0.08).

Columnas: Original | Nuevo (64 pts) | DIP ~82 pts (solo g=-4,-2,-0.5) | DIP ~164 pts |
DIP ~328 pts. Dos figuras con la misma grilla: reconstrucciones y |error|.
PSNR recalculado igual para todos (128x128, sin el pad de DIP, contra sim_128.png).

Ojo al comparar: frontier_mix usa el campo denso para ubicar sus puntos (oraculo); el
metodo nuevo solo consulta puntos (desplegable con KMC).

Uso:  ./venv/bin/python -m analysis.comparacion_nuevo_vs_dip
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio

from autoexp.oracle import load_truth

MAIN = os.environ.get("MAIN", "/home/santiafonso/Tesis")
NEW = os.environ.get("NEW", "autoexp/runs/VAL9_loo_n30_off0.5")
OUT = os.environ.get("OUT", "results/comparacion_nuevo_vs_dip")
GS = ["-4.0", "-3.5", "-3.0", "-2.5", "-2.0", "-1.5", "-1.0", "-0.5", "0.0"]
EXP = os.path.join(MAIN, "results/experimentos_menos_puntos")


def crop(r):
    r = np.load(r).astype(float)
    r = r[0] if r.ndim == 3 else r
    return np.clip(r[1:-1, 1:-1] if r.shape == (130, 130) else r, 0, 1)


def n_mask(g, mf):
    p = os.path.join(MAIN, "data/restoration/masks_g/g%s/mf%s/mask_frontier_mix.npy" % (g, mf))
    return int(np.load(p).sum()) if os.path.exists(p) else None


def sources(g):
    out = [("Nuevo (sin DIP)", os.path.join(NEW, "g%s/restored.npy" % g), 64)]
    out.append(("DIP frontier_mix", os.path.join(EXP, "fewpoints_3g/g%s/mf0.995/frontier_mix/restored.npy" % g),
                n_mask(g, "0.995")))
    for mf in ("0.990", "0.980"):
        out.append(("DIP frontier_mix", os.path.join(
            EXP, "barrido_reg_noise_familias/g%s/mf%s/frontier_mix/reg0.08/restored.npy" % (g, mf)), n_mask(g, mf)))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    ncol = 5
    figs = {k: plt.subplots(len(GS), ncol, figsize=(2.6 * ncol, 2.6 * len(GS))) for k in ("rec", "err")}
    heads = ["Original", "Nuevo: 64 pts\n(sin DIP)", "DIP ~82 pts", "DIP ~164 pts", "DIP ~328 pts"]
    table = []
    for r, g in enumerate(GS):
        t = load_truth(g)
        row = [g]
        for k in figs:
            ax = figs[k][1][r, 0]
            ax.imshow(t, cmap="viridis", vmin=0, vmax=1)
            ax.set_ylabel("g = %s" % g, fontsize=11)
        for c, (name, path, npts) in enumerate(sources(g), start=1):
            axr, axe = figs["rec"][1][r, c], figs["err"][1][r, c]
            if not os.path.exists(path):
                for ax in (axr, axe):
                    ax.text(0.5, 0.5, "no corrido", ha="center", va="center", color="0.5", fontsize=9,
                            transform=ax.transAxes)
                row.append("")
                continue
            rec = crop(path)
            ps = peak_signal_noise_ratio(t, rec, data_range=1.0)
            row.append("%.1f" % ps)
            lab = "%.1f dB" % ps + ("  (%d pts)" % npts if npts and c > 1 else "")
            axr.imshow(rec, cmap="viridis", vmin=0, vmax=1)
            axe.imshow(np.abs(rec - t), cmap="magma", vmin=0, vmax=0.3)
            for ax in (axr, axe):
                ax.set_title(lab, fontsize=9, color="C3" if c == 1 else "k", fontweight="bold" if c == 1 else None)
        table.append(row)
    for k, (fig, ax) in figs.items():
        for a in ax.ravel():
            a.set_xticks([])
            a.set_yticks([])
        for c, h in enumerate(heads):
            ax[0, c].annotate(h, (0.5, 1.28), xycoords="axes fraction", ha="center", va="bottom", fontsize=11)
        fig.suptitle(("Reconstrucciones" if k == "rec" else "|error| (escala 0 - 0.3)")
                     + ": metodo nuevo con 64 puntos vs mejor DIP previo (frontier_mix + reg 0.08)",
                     fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.975))
        fig.savefig(os.path.join(OUT, "comparacion_%s.png" % k), dpi=90)
    with open(os.path.join(OUT, "psnr.csv"), "w") as f:
        f.write("g,nuevo_64,dip_82,dip_164,dip_328\n")
        for row in table:
            f.write(",".join(row) + "\n")
    print(open(os.path.join(OUT, "psnr.csv")).read())


if __name__ == "__main__":
    main()
